"""
Scanner module for detecting libraries and frameworks in a project.

This module scans a project directory to identify which libraries and frameworks
are being used based on package manager files, import statements, and
framework-specific file patterns.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from cursor_rules_cli import utils

logger = logging.getLogger(__name__)

# File patterns to look for
PACKAGE_PATTERNS = {
    "node": [
        "package.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "package-lock.json"
    ],
    "python": [
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "setup.py",
        "uv.lock",
        "poetry.lock",
        "conda.yaml",
        "environment.yml"
    ],
    "php": ["composer.json", "composer.lock"],
    "rust": ["Cargo.toml", "Cargo.lock"],
    "go": ["go.mod", "go.sum"],
    "ruby": ["Gemfile", "Gemfile.lock"],
    "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
    "dotnet": ["*.csproj", "*.fsproj", "*.vbproj", "packages.config"],
}

# Framework-specific file patterns
FRAMEWORK_PATTERNS = {
    "react": ["src/App.jsx", "src/App.tsx", "src/App.js", "public/index.html"],
    "vue": ["src/App.vue", "src/main.js", "public/index.html"],
    "angular": ["angular.json", "src/app/app.module.ts"],
    "next-js": ["next.config.js", "pages/_app.js", "pages/_app.tsx"],
    "nuxt": ["nuxt.config.js", "nuxt.config.ts"],
    "svelte": ["svelte.config.js", "src/App.svelte"],
    "django": ["manage.py", "wsgi.py", "asgi.py"],
    "flask": ["app.py", "wsgi.py", "application.py"],
    "fastapi": ["main.py"],
    "express": ["app.js", "server.js"],
    "nestjs": ["nest-cli.json", "src/main.ts"],
    "laravel": ["artisan", "composer.json"],
    "spring-boot": ["src/main/java", "src/main/resources/application.properties"],
}

# Import patterns for different languages
IMPORT_PATTERNS = {
    "python": {
        "files": ["*.py"],
        "regex": [
            r"(?:^|\n)\s*(?:import|from)\s+([a-zA-Z0-9_.]+)",
            r"(?:^|\n)\s*from\s+([a-zA-Z0-9_.]+)\s+import",
            r"(?:^|\n)\s*__import__\(['\"]([a-zA-Z0-9_.]+)['\"]\)",
            r"(?:^|\n)\s*importlib\.import_module\(['\"]([a-zA-Z0-9_.]+)['\"]\)"
        ]
    },
    "javascript": {
        "files": ["*.js", "*.jsx", "*.ts", "*.tsx"],
        "regex": [
            r"(?:^|\n)\s*import\s+.*?(?:from\s+['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"])",
            r"(?:^|\n)\s*require\(['\"]([^'\"]+)['\"]\)",
            r"(?:^|\n)\s*import\(['\"]([^'\"]+)['\"]\)"
        ]
    },
    "php": {
        "files": ["*.php"],
        "regex": [
            r"(?:^|\n)\s*(?:use|require|include|require_once|include_once)\s+['\"]?([a-zA-Z0-9_\\/.]+)",
            r"(?:^|\n)\s*namespace\s+([a-zA-Z0-9_\\/.]+)"
        ]
    },
    "java": {
        "files": ["*.java"],
        "regex": [
            r"(?:^|\n)\s*import\s+([a-zA-Z0-9_.]+)",
            r"(?:^|\n)\s*package\s+([a-zA-Z0-9_.]+)"
        ]
    },
    "rust": {
        "files": ["*.rs"],
        "regex": [
            r"(?:^|\n)\s*(?:use|extern\s+crate)\s+([a-zA-Z0-9_:]+)",
            r"(?:^|\n)\s*mod\s+([a-zA-Z0-9_]+)"
        ]
    },
}

# Directories to exclude from scanning
EXCLUDED_DIRS = [
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "target",
    "out",
    "bin",
    "obj",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "site-packages",
    "lib/python*",
]

# Maximum directory depth for import scanning
MAX_SCAN_DEPTH = 5

def scan_project(
    project_dir: str,
    quick_scan: bool = False,
    max_depth: int = MAX_SCAN_DEPTH,
    rules_path: str = None,
    max_workers: int = None,
    use_cache: bool = True
) -> List[str]:
    """
    Scan a project directory to detect libraries and frameworks.
    
    Args:
        project_dir: Path to the project directory
        quick_scan: If True, only scan package files, not imports
        max_depth: Maximum directory depth for scanning
        rules_path: Path to rules.json file
        max_workers: Maximum number of worker threads (None for CPU count)
        use_cache: Whether to use caching
        
    Returns:
        List of detected libraries and frameworks
    """
    project_path = Path(project_dir).resolve()
    logger.debug(f"Scanning project at {project_path}")
    
    # Check cache first
    if use_cache:
        cache_key = utils.create_cache_key(
            str(project_path),
            quick_scan,
            max_depth,
            rules_path
        )
        cached_data = utils.get_cached_data(cache_key)
        if cached_data:
            logger.debug("Using cached scan results")
            return cached_data
    
    # Load library data from rules.json
    library_data = utils.load_library_data(rules_path)
    
    # Track both the libraries and their sources
    detected_libraries = set()
    direct_match_libraries = set()  # Track direct matches separately
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit package file scanning first to identify direct dependencies
        package_files_future = executor.submit(scan_package_files, project_path)
        
        # Submit other scanning tasks
        future_to_task = {
            executor.submit(scan_docker_files, project_path): "docker_files",
            executor.submit(scan_github_actions, project_path): "github_actions",
            executor.submit(detect_frameworks, project_path): "frameworks"
        }
        
        # Process package files result first to identify direct dependencies
        try:
            direct_matches = package_files_future.result()
            detected_libraries.update(direct_matches)
            direct_match_libraries.update(direct_matches)  # Mark as direct matches
            logger.debug(f"Completed package_files scan, found {len(direct_matches)} direct dependencies")
        except Exception as e:
            logger.error(f"Error in package_files scan: {e}")
        
        # Add import scanning if not quick scan
        if not quick_scan:
            future_to_task[executor.submit(scan_imports, project_path, max_depth)] = "imports"
        
        # Process results from other scanning tasks
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                detected_libraries.update(result)
                logger.debug(f"Completed {task_name} scan")
            except Exception as e:
                logger.error(f"Error in {task_name} scan: {e}")
    
    # Normalize library names
    normalized_libraries = {
        utils.normalize_library_name(lib, library_data)
        for lib in detected_libraries
    }
    
    normalized_direct_matches = {
        utils.normalize_library_name(lib, library_data)
        for lib in direct_match_libraries
    }
    
    # Detect additional frameworks based on rules.json
    if library_data:
        framework_libs = detect_frameworks_from_rules(normalized_libraries, library_data)
        normalized_libraries.update(framework_libs)
    
    # Sort libraries, prioritizing direct matches first, then by popularity
    sorted_libraries = sorted(
        normalized_libraries,
        key=lambda x: (
            x in normalized_direct_matches,  # Direct matches first
            utils.calculate_library_popularity(x, library_data)  # Then by popularity
        ),
        reverse=True
    )
    
    # Cache results
    if use_cache:
        utils.set_cached_data(cache_key, sorted_libraries)
    
    return sorted_libraries

def scan_package_files(project_path: Path) -> Set[str]:
    """
    Scan package manager files to detect libraries.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Set of detected libraries
    """
    detected_libs = set()
    
    # Check for Node.js package files
    for node_file in ["package.json", "yarn.lock", "pnpm-lock.yaml"]:
        file_path = project_path / node_file
        if file_path.exists():
            logger.debug(f"Found {node_file}")
            try:
                if node_file == "package.json":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Add dependencies
                    deps = data.get("dependencies", {})
                    dev_deps = data.get("devDependencies", {})
                    all_deps = {**deps, **dev_deps}
                    
                    # Add detected libraries
                    detected_libs.update(all_deps.keys())
                    
                    # Detect framework from dependencies
                    framework_deps = {
                        "react": "react",
                        "vue": "vue",
                        "next": "next-js",
                        "nuxt": "nuxt",
                        "svelte": "svelte",
                        "@angular/core": "angular",
                        "express": "express",
                        "@nestjs/core": "nestjs"
                    }
                    
                    for dep, framework in framework_deps.items():
                        if dep in deps:
                            detected_libs.add(framework)
                
                elif node_file == "yarn.lock":
                    with open(file_path, 'r') as f:
                        content = f.read()
                    # Extract package names from yarn.lock
                    packages = re.findall(r'^"?([^@\s"]+)@', content, re.MULTILINE)
                    detected_libs.update(packages)
                
                elif node_file == "pnpm-lock.yaml":
                    with open(file_path, 'r') as f:
                        content = f.read()
                    # Extract package names from pnpm-lock.yaml
                    packages = re.findall(r'(?:^|\n)\s*/([^/:]+):', content)
                    detected_libs.update(packages)
                    
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error parsing {node_file}: {e}")
    
    # Check for Python package files
    python_files = {
        "requirements.txt": r'^([a-zA-Z0-9_.-]+)',
        "pyproject.toml": None,  # Pattern not needed, handled specially
        "Pipfile": r'(?:^|\n)\s*([a-zA-Z0-9_.-]+)\s*=',
        "setup.py": r'install_requires=\[([^\]]+)\]',
        # Keep uv.lock for compatibility with uv (modern Python package manager)
        # Only check for it if it exists to avoid unnecessary file operations
        "uv.lock": r'name\s*=\s*"([^"]+)"' if os.path.exists(project_path / "uv.lock") else None
    }
    
    for file_name, pattern in python_files.items():
        file_path = project_path / file_name
        if file_path.exists():
            logger.debug(f"Found {file_name}")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if file_name == "setup.py":
                    # Special handling for setup.py
                    matches = re.search(pattern, content)
                    if matches:
                        packages = re.findall(r'[\'"]([^\'\"]+)[\'"]', matches.group(1))
                        detected_libs.update(p.split('>=')[0].split('==')[0].strip() for p in packages)
                elif file_name == "pyproject.toml":
                    # Special handling for pyproject.toml
                    # Look for dependencies section in PEP 621 format
                    pep621_deps_match = re.search(r'\[project\].*?dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if pep621_deps_match:
                        deps_content = pep621_deps_match.group(1)
                        # Extract package names from dependencies
                        packages = re.findall(r'[\'"]([a-zA-Z0-9_.-]+)(?:>=|==|>|<|~=|!=|@|$)', deps_content)
                        detected_libs.update(packages)
                    
                    # Look for dependencies section in Poetry format
                    poetry_deps_match = re.search(r'\[tool\.poetry\.dependencies\](.*?)(?:\[|\Z)', content, re.DOTALL)
                    if poetry_deps_match:
                        deps_content = poetry_deps_match.group(1)
                        # Extract package names from Poetry dependencies
                        packages = re.findall(r'([a-zA-Z0-9_.-]+)\s*=', deps_content)
                        detected_libs.update(packages)
                    
                    # Also check for dev-dependencies in Poetry format
                    dev_deps_match = re.search(r'\[tool\.poetry\.dev-dependencies\](.*?)(?:\[|\Z)', content, re.DOTALL)
                    if dev_deps_match:
                        dev_deps_content = dev_deps_match.group(1)
                        dev_packages = re.findall(r'([a-zA-Z0-9_.-]+)\s*=', dev_deps_content)
                        detected_libs.update(dev_packages)
                else:
                    # General pattern matching
                    packages = re.findall(pattern, content)
                    detected_libs.update(p.split('>=')[0].split('==')[0].strip() for p in packages)
                    
                # Check for common frameworks
                framework_packages = {"django", "flask", "fastapi"}
                detected_libs.update(framework_packages & detected_libs)
                
            except IOError as e:
                logger.warning(f"Error reading {file_name}: {e}")
    
    return detected_libs

def scan_docker_files(project_path: Path) -> Set[str]:
    """
    Scan Dockerfile and docker-compose files for libraries.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Set of detected libraries
    """
    detected_libs = set()
    
    docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
    for file_name in docker_files:
        file_path = project_path / file_name
        if file_path.exists():
            logger.debug(f"Found {file_name}")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Look for common package installations
                pip_packages = re.findall(r'pip\s+install\s+([^\s&|;]+)', content)
                npm_packages = re.findall(r'npm\s+install\s+([^\s&|;]+)', content)
                apt_packages = re.findall(r'apt-get\s+install\s+([^\s&|;]+)', content)
                
                detected_libs.update(pip_packages)
                detected_libs.update(npm_packages)
                detected_libs.update(apt_packages)
                
                # Look for base images
                base_images = re.findall(r'FROM\s+([^\s:]+)', content)
                detected_libs.update(base_images)
                
            except IOError as e:
                logger.warning(f"Error reading {file_name}: {e}")
    
    return detected_libs

def scan_github_actions(project_path: Path) -> Set[str]:
    """
    Scan GitHub Actions workflow files for libraries.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Set of detected libraries
    """
    detected_libs = set()
    
    workflows_dir = project_path / ".github" / "workflows"
    if not workflows_dir.exists():
        return detected_libs
    
    for workflow_file in workflows_dir.glob("*.yml"):
        logger.debug(f"Found workflow file: {workflow_file}")
        try:
            with open(workflow_file, 'r') as f:
                content = f.read()
            
            # Look for common actions and tools
            actions = re.findall(r'uses:\s+([^\s@]+)', content)
            detected_libs.update(actions)
            
            # Look for package installations
            pip_packages = re.findall(r'pip\s+install\s+([^\s&|;]+)', content)
            npm_packages = re.findall(r'npm\s+install\s+([^\s&|;]+)', content)
            
            detected_libs.update(pip_packages)
            detected_libs.update(npm_packages)
            
        except IOError as e:
            logger.warning(f"Error reading workflow file {workflow_file}: {e}")
    
    return detected_libs

def detect_frameworks(project_path: Path) -> Set[str]:
    """
    Detect frameworks based on specific file patterns.
    
    Args:
        project_path: Path to the project directory
        
    Returns:
        Set of detected frameworks
    """
    detected_frameworks = set()
    
    for framework, patterns in FRAMEWORK_PATTERNS.items():
        for pattern in patterns:
            # Check if the pattern is a directory
            if not pattern.endswith(('/', '\\')) and not os.path.splitext(pattern)[1]:
                if (project_path / pattern).is_dir():
                    logger.debug(f"Found framework directory pattern: {pattern}")
                    detected_frameworks.add(framework)
                    break
            
            # Check for file patterns
            matches = list(project_path.glob(pattern))
            if matches:
                logger.debug(f"Found framework file pattern: {pattern}")
                detected_frameworks.add(framework)
                break
    
    return detected_frameworks

def detect_frameworks_from_rules(detected_libs: Set[str], library_data: Dict[str, Any]) -> Set[str]:
    """
    Detect frameworks based on detected libraries and rules.json data.
    
    Args:
        detected_libs: Set of detected libraries
        library_data: Library data from rules.json
        
    Returns:
        Set of detected frameworks
    """
    detected_frameworks = set()
    
    if not library_data or "libraries" not in library_data:
        return detected_frameworks
    
    # Create a mapping of library names to their data
    lib_map = {lib["name"].lower(): lib for lib in library_data["libraries"]}
    
    # Check detected libraries against rules.json
    for lib in detected_libs:
        lib_lower = lib.lower()
        if lib_lower in lib_map:
            # Add the library itself
            detected_frameworks.add(lib_lower)
            
            # Check if this library is a framework
            tags = lib_map[lib_lower].get("tags", [])
            if "framework" in tags:
                detected_frameworks.add(lib_lower)
                
            # Check for related libraries based on tags
            # For example, if we detect "react", we might want to check for "react-router"
            if "react" in lib_lower and "frontend" in tags:
                for related_lib, related_data in lib_map.items():
                    if "react" in related_lib and related_lib != lib_lower:
                        if any(tag in related_data.get("tags", []) for tag in ["frontend", "ui"]):
                            detected_frameworks.add(related_lib)
    
    return detected_frameworks

def scan_imports(project_path: Path, max_depth: int = MAX_SCAN_DEPTH) -> Set[str]:
    """
    Scan source files for import statements to detect libraries.
    
    Args:
        project_path: Path to the project directory
        max_depth: Maximum directory depth for scanning
        
    Returns:
        Set of detected libraries from imports
    """
    detected_imports = set()
    
    for lang, pattern_info in IMPORT_PATTERNS.items():
        file_patterns = pattern_info["files"]
        import_regexes = pattern_info["regex"]
        
        # Use a more efficient file traversal with depth limit and exclusions
        for file_pattern in file_patterns:
            for file_path in find_files(project_path, file_pattern, max_depth):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Find all imports using multiple regex patterns
                    for import_regex in import_regexes:
                        imports = re.findall(import_regex, content)
                        
                        # Process matches
                        for imp in imports:
                            if isinstance(imp, tuple):
                                # Some regex patterns might have multiple capture groups
                                imp = next((i for i in imp if i), "")
                            
                            if imp:
                                # Extract the top-level package name
                                top_level = imp.split('.')[0].split('/')[0]
                                if top_level and not top_level.startswith(('.', '_')):
                                    detected_imports.add(top_level.lower())
                
                except (IOError, UnicodeDecodeError) as e:
                    logger.debug(f"Error reading {file_path}: {e}")
    
    return detected_imports

def find_files(root_dir: Path, pattern: str, max_depth: int, current_depth: int = 0) -> List[Path]:
    """
    Find files matching a pattern with depth limit and directory exclusions.
    
    Args:
        root_dir: Root directory to start searching from
        pattern: File pattern to match
        max_depth: Maximum directory depth to search
        current_depth: Current depth in the directory tree
        
    Returns:
        List of file paths matching the pattern
    """
    if current_depth > max_depth:
        return []
    
    matching_files = []
    
    try:
        for item in root_dir.iterdir():
            if item.is_file() and item.match(pattern):
                matching_files.append(item)
            elif item.is_dir() and not should_exclude_dir(item):
                matching_files.extend(find_files(item, pattern, max_depth, current_depth + 1))
    except (PermissionError, OSError) as e:
        logger.debug(f"Error accessing {root_dir}: {e}")
    
    return matching_files

def should_exclude_dir(dir_path: Path) -> bool:
    """
    Check if a directory should be excluded from scanning.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        True if the directory should be excluded, False otherwise
    """
    dir_name = dir_path.name
    return dir_name in EXCLUDED_DIRS or dir_name.startswith('.')

if __name__ == "__main__":
    # For testing
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "."
        
    libraries = scan_project(project_dir)
    print(f"Detected libraries: {libraries}") 