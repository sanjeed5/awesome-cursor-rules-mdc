"""
Matcher module for matching detected libraries with MDC rules.

This module matches detected libraries with MDC rules based on relevance scores,
library relationships, and project context.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from cursor_rules_cli import utils

logger = logging.getLogger(__name__)

# Minimum relevance score for a rule to be considered
MIN_RELEVANCE_SCORE = 0.5

# Maximum number of rules to return
MAX_RULES = 10

def match_libraries(
    detected_libraries: List[str],
    source_url: str,
    direct_match_libraries: Optional[Set[str]] = None,
    custom_json_path: Optional[Path] = None,
    max_results: int = MAX_RULES,
    min_score: float = MIN_RELEVANCE_SCORE
) -> List[Dict[str, Any]]:
    """
    Match detected libraries with available rules.
    
    Args:
        detected_libraries: List of detected libraries
        source_url: Base URL for the repository
        direct_match_libraries: Set of libraries that are direct matches from package files
        custom_json_path: Path to custom rules.json file
        max_results: Maximum number of rules to return
        min_score: Minimum relevance score for rules
        
    Returns:
        List of matched rules with metadata
    """
    # Create a RuleMatcher instance
    matcher = RuleMatcher(
        rules_path=str(custom_json_path) if custom_json_path else None,
        min_relevance_score=min_score,
        max_rules=max_results
    )
    
    # Match rules
    matched_rules = matcher.match_rules(detected_libraries)
    
    # Add URL and other metadata to each rule
    for rule in matched_rules:
        rule_name = rule.get("rule")
        rule["name"] = rule_name
        # We don't construct a URL here anymore - the downloader will handle this using GitHub API
        
        # Mark direct matches
        if direct_match_libraries and rule_name.lower() in (lib.lower() for lib in direct_match_libraries):
            rule["is_direct_match"] = True
        else:
            rule["is_direct_match"] = False
        
    return matched_rules

class RuleMatcher:
    """
    Class for matching detected libraries with MDC rules.
    """
    
    def __init__(
        self,
        rules_path: str = None,
        use_cache: bool = True,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
        max_rules: int = MAX_RULES
    ):
        """
        Initialize the RuleMatcher.
        
        Args:
            rules_path: Path to rules.json file
            use_cache: Whether to use caching
            min_relevance_score: Minimum relevance score for a rule
            max_rules: Maximum number of rules to return
        """
        self.rules_path = rules_path
        self.use_cache = use_cache
        self.min_relevance_score = min_relevance_score
        self.max_rules = max_rules
        
        # Load library data from rules.json
        self.library_data = utils.load_library_data(rules_path)
        
        # Create library mappings
        self._create_library_mappings()
    
    def _create_library_mappings(self):
        """Create mappings for efficient library lookups."""
        self.lib_to_tags = {}
        self.tag_to_libs = {}
        self.lib_to_related = {}
        
        if not self.library_data or "libraries" not in self.library_data:
            return
        
        for lib in self.library_data["libraries"]:
            lib_name = lib["name"].lower()
            
            # Map library to its tags
            tags = lib.get("tags", [])
            self.lib_to_tags[lib_name] = set(tags)
            
            # Map tags to libraries
            for tag in tags:
                if tag not in self.tag_to_libs:
                    self.tag_to_libs[tag] = set()
                self.tag_to_libs[tag].add(lib_name)
            
            # Map library to related libraries
            related = lib.get("related", [])
            self.lib_to_related[lib_name] = set(related)
    
    def match_rules(
        self,
        detected_libraries: List[str],
        project_context: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Match detected libraries with MDC rules.
        
        Args:
            detected_libraries: List of detected libraries
            project_context: Optional project context scores
            
        Returns:
            List of matched rules with relevance scores
        """
        if not self.library_data or "libraries" not in self.library_data:
            logger.warning("No libraries found in rules.json")
            return []
        
        # Check cache first
        if self.use_cache:
            cache_key = utils.create_cache_key(
                ",".join(sorted(detected_libraries)),
                str(project_context),
                self.min_relevance_score,
                self.max_rules
            )
            cached_data = utils.get_cached_data(cache_key)
            if cached_data:
                logger.debug("Using cached rule matches")
                return cached_data
        
        # Normalize library names
        normalized_libs = {
            utils.normalize_library_name(lib, self.library_data)
            for lib in detected_libraries
        }
        
        # Get project context if not provided
        if project_context is None:
            project_context = utils.get_project_context(normalized_libs, self.library_data)
        
        # Calculate relevance scores for each library in rules.json
        library_scores = []
        for library in self.library_data["libraries"]:
            score = self._calculate_library_relevance(
                library,
                normalized_libs,
                project_context
            )
            
            if score >= self.min_relevance_score:
                library_scores.append((library, score))
        
        # Sort libraries by relevance score
        library_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for library, score in library_scores[:self.max_rules]:
            result = {
                "rule": library["name"],
                "relevance_score": round(score, 3),
                "description": f"{library['name']} ({', '.join(library.get('tags', []))})",
                "tags": library.get("tags", []),
                "libraries": [library["name"]],
                "category": self._categorize_library(library, normalized_libs)
            }
            results.append(result)
        
        # Cache results
        if self.use_cache:
            utils.set_cached_data(cache_key, results)
        
        return results
    
    def _calculate_library_relevance(
        self,
        library: Dict[str, Any],
        detected_libs: Set[str],
        project_context: Dict[str, float]
    ) -> float:
        """
        Calculate relevance score for a library.
        
        Args:
            library: Library data
            detected_libs: Set of detected libraries
            project_context: Project context scores
            
        Returns:
            Relevance score between 0 and 1
        """
        # Direct match score
        lib_name = library["name"].lower()
        direct_match = 1.0 if lib_name in detected_libs else 0.0
        
        # Tag similarity score
        tag_score = self._calculate_tag_similarity_score(library, detected_libs)
        
        # Context score from project type and tags
        context_score = self._calculate_context_score(library, project_context)
        
        # Combine scores with weights
        weights = {
            "direct_match": 0.8,
            "tag_similarity": 0.15,
            "context": 0.05
        }
        
        total_score = (
            weights["direct_match"] * direct_match +
            weights["tag_similarity"] * tag_score +
            weights["context"] * context_score
        )
        
        return total_score
    
    def _calculate_context_score(
        self,
        library: Dict[str, Any],
        project_context: Dict[str, float]
    ) -> float:
        """
        Calculate context match score for a library.
        
        Args:
            library: Library data
            project_context: Project context scores
            
        Returns:
            Score between 0 and 1
        """
        library_tags = set(library.get("tags", []))
        if not library_tags or not project_context:
            return 0
        
        # Calculate weighted average of context scores for matching tags
        total_score = 0
        total_weight = 0
        
        for tag in library_tags:
            if tag in project_context:
                weight = project_context[tag]
                total_score += weight
                total_weight += 1
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_tag_similarity_score(
        self,
        library: Dict[str, Any],
        detected_libs: Set[str]
    ) -> float:
        """
        Calculate tag similarity score for a library.
        
        Args:
            library: Library data
            detected_libs: Set of detected libraries
            
        Returns:
            Score between 0 and 1
        """
        library_tags = set(library.get("tags", []))
        if not library_tags:
            return 0
        
        # Get all tags from detected libraries
        lib_tags = set()
        for lib in detected_libs:
            if lib in self.lib_to_tags:
                lib_tags.update(self.lib_to_tags[lib])
        
        if not lib_tags:
            return 0
        
        # Calculate Jaccard similarity
        intersection = library_tags & lib_tags
        union = library_tags | lib_tags
        
        return len(intersection) / len(union)
    
    def _categorize_library(
        self,
        library: Dict[str, Any],
        detected_libs: Set[str]
    ) -> str:
        """
        Categorize a library based on its relationship to detected libraries.
        
        Args:
            library: Library data
            detected_libs: Set of detected libraries
            
        Returns:
            Category string
        """
        lib_name = library["name"].lower()
        
        # Check for direct matches
        if lib_name in detected_libs:
            return "direct_match"
        
        # Check for tag matches
        library_tags = set(library.get("tags", []))
        lib_tags = set()
        for lib in detected_libs:
            if lib in self.lib_to_tags:
                lib_tags.update(self.lib_to_tags[lib])
        
        if library_tags & lib_tags:
            return "tag_match"
        
        return "suggested"

if __name__ == "__main__":
    # For testing
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) > 1:
        rules_path = sys.argv[1]
    else:
        rules_path = None
    
    # Example usage
    matcher = RuleMatcher(rules_path)
    detected_libs = ["react", "next-js", "tailwindcss"]
    matched_rules = matcher.match_rules(detected_libs)
    
    print("\nDetected libraries:", detected_libs)
    print("\nMatched rules:")
    for rule in matched_rules:
        print(f"\n{rule['rule']} (score: {rule['relevance_score']}):")
        print(f"  Category: {rule['category']}")
        print(f"  Description: {rule['description']}")
        print(f"  Tags: {', '.join(rule['tags'])}")
        print(f"  Libraries: {', '.join(rule['libraries'])}") 