---
description: This rule enforces best practices and coding standards for Zsh scripting to enhance readability, maintainability, security, and performance. It covers code structure, common patterns, performance, security, testing, common pitfalls, and tooling.
globs: *.zsh
---
# Zsh Best Practices and Coding Standards

This document outlines the best practices and coding standards for Zsh scripting, aiming to enhance readability, maintainability, security, and performance of your scripts.

## 1. Code Organization and Structure

*   **Directory Structure Best Practices:**
    *   `bin/`: Executable scripts intended for general use.
    *   `lib/`: Library functions and modules.
    *   `conf/`: Configuration files.
    *   `test/`: Test scripts.
    *   `modules/`: Dynamically loaded modules.
    *   `data/`: Data files used by scripts.
    *   `tmp/`: Temporary files (ensure proper cleanup).
*   **File Naming Conventions:**
    *   Use `.zsh` extension for Zsh scripts.
    *   Descriptive names using lowercase and underscores (e.g., `process_data.zsh`, `utility_functions.zsh`).
    *   Avoid spaces in file names.
*   **Module Organization:**
    *   Split code into logical modules (e.g., `network_utils.zsh`, `string_manipulation.zsh`).
    *   Use `source` or `.` to include modules in your scripts.
        zsh
        source lib/network_utils.zsh
        
    *   Consider using Zsh's module loading capabilities for larger projects.
        zsh
        zmodload zsh/datetime
        
*   **Component Architecture:**
    *   Design scripts with modular, reusable components.
    *   Encapsulate functionality within functions.
    *   Use meaningful function names that describe their purpose.
*   **Code Splitting Strategies:**
    *   Split large scripts into smaller, manageable files.
    *   Group related functions into modules.
    *   Use functions to abstract complex logic.

## 2. Common Patterns and Anti-patterns

*   **Design Patterns:**
    *   **Command Pattern:** Encapsulate commands as objects.
    *   **Strategy Pattern:** Define a family of algorithms, encapsulate each one, and make them interchangeable.
    *   **Template Method Pattern:** Define the skeleton of an algorithm in a function, deferring some steps to sub-functions.
*   **Recommended Approaches for Common Tasks:**
    *   **Parsing Command-Line Arguments:** Use `getopts` or `zparseopts` for parsing command-line arguments.
        zsh
        while getopts 'a:b:c' opt;
        do
          case "$opt" in
            a) arg_a="$OPTARG" ;;
            b) arg_b="$OPTARG" ;;
            c) flag_c=true ;;
            \?) echo "Invalid option: -$OPTARG" >&2 ;;
          esac
        done
        shift "$((OPTIND - 1))"
        
    *   **String Manipulation:** Use Zsh's built-in string manipulation features (e.g., parameter expansion, pattern matching).
    *   **File I/O:** Use redirection (`>`, `<`, `>>`) and pipes (`|`) for file input/output.
*   **Anti-patterns and Code Smells:**
    *   **Global Variables:** Avoid global variables; use local variables within functions.
    *   **Magic Numbers:** Avoid hardcoding values directly; use named constants.
    *   **Code Duplication:** Refactor duplicated code into reusable functions or modules.
    *   **Long Functions:** Break down long functions into smaller, more manageable units.
    *   **Unclear Error Messages:** Provide informative error messages to the user.
*   **State Management:**
    *   Minimize the use of global state.
    *   Pass state as arguments to functions.
    *   Use temporary files for persistent state (with proper cleanup).
*   **Error Handling:**
    *   Use `set -e` to exit immediately if a command exits with a non-zero status.
    *   Use `set -u` to treat unset variables as an error.
    *   Check the exit status of commands using `$?`.
    *   Use `trap` to handle signals (e.g., `EXIT`, `INT`, `TERM`).
        zsh
        trap 'echo "Script interrupted."; exit 1' INT TERM
        
    *   Redirect stderr to a file for debugging.

## 3. Performance Considerations

*   **Optimization Techniques:**
    *   **Avoid Spawning External Processes:** Use Zsh's built-in commands whenever possible.
    *   **Minimize Disk I/O:** Use caching to reduce disk access.
    *   **Optimize Loops:** Use efficient looping constructs (e.g., `for` loops over arrays).
    *   **Use Arrays:** Arrays are generally faster than string manipulation for certain tasks.
    *   **Use `zcompile`:** Pre-compile your zsh scripts to improve startup time
        zsh
        zcompile script.zsh
        
*   **Memory Management:**
    *   Be mindful of memory usage when working with large files or data sets.
    *   Use `unset` to free memory occupied by variables that are no longer needed.
*   **Rendering Optimization (if applicable):**
    *   Optimize terminal output for speed and clarity.
    *   Avoid excessive use of colors and formatting.
*   **Bundle Size Optimization:**
    *   Not applicable to Zsh scripts in the same way as web applications, but keep scripts concise and avoid unnecessary dependencies.
*   **Lazy Loading:**
    *   Load modules and functions only when they are needed.

## 4. Security Best Practices

*   **Common Vulnerabilities:**
    *   **Command Injection:** Prevent command injection by properly quoting variables and avoiding the use of `eval`.
    *   **Path Traversal:** Sanitize user input to prevent path traversal attacks.
    *   **Denial of Service (DoS):** Limit resource consumption to prevent DoS attacks.
*   **Input Validation:**
    *   Validate all user input before using it in commands.
    *   Use regular expressions to validate data formats.
    *   Use `typeset -i` to ensure variables are integers when expected.
*   **Authentication and Authorization:**
    *   For scripts requiring authentication, consider using existing authentication mechanisms (e.g., SSH keys).
    *   Implement authorization checks to restrict access to sensitive resources.
*   **Data Protection:**
    *   Store sensitive data securely (e.g., using encrypted files).
    *   Avoid storing passwords directly in scripts.
    *   Use environment variables for sensitive configuration information.
*   **Secure API Communication:**
    *   Use HTTPS for all API communication.
    *   Validate API responses.
    *   Handle API errors gracefully.

## 5. Testing Approaches

*   **Unit Testing:**
    *   Test individual functions in isolation.
    *   Use a testing framework like `zunit` or `zrtest`.
    *   Write tests for both positive and negative scenarios.
*   **Integration Testing:**
    *   Test the interaction between different modules and components.
    *   Simulate real-world scenarios.
*   **End-to-End Testing:**
    *   Test the entire script from start to finish.
    *   Verify that the script produces the expected output.
*   **Test Organization:**
    *   Create a separate directory for test scripts.
    *   Organize tests by module or component.
*   **Mocking and Stubbing:**
    *   Use mocking and stubbing to isolate units of code during testing.
    *   Create mock objects for external dependencies.

## 6. Common Pitfalls and Gotchas

*   **Frequent Mistakes:**
    *   **Forgetting to Quote Variables:** Always quote variables to prevent word splitting and globbing.
    *   **Ignoring Exit Status:** Always check the exit status of commands.
    *   **Using `eval` Unnecessarily:** Avoid `eval` if possible, as it can lead to security vulnerabilities.
    *   **Not Handling Signals:** Handle signals like `INT` and `TERM` to ensure proper cleanup.
*   **Edge Cases:**
    *   Handling empty or missing input.
    *   Dealing with special characters in filenames.
    *   Handling different locales and character encodings.
*   **Version-Specific Issues:**
    *   Be aware of differences between Zsh versions.
    *   Use feature detection to ensure compatibility.
*   **Compatibility Concerns:**
    *   Ensure that your scripts are compatible with different operating systems.
    *   Test your scripts on different environments.
*   **Debugging Strategies:**
    *   Use `set -x` to trace the execution of your script.
    *   Use `echo` statements to print debugging information.
    *   Use a debugger like `gdb` or `lldb` (if available).

## 7. Tooling and Environment

*   **Recommended Development Tools:**
    *   **Text Editor:** Use a text editor with Zsh syntax highlighting and linting (e.g., VS Code with Zsh extensions).
    *   **Shellcheck:** Use Shellcheck to identify potential problems in your scripts.
    *   **Zsh Debugger:** If available, use a debugger to step through your code.
*   **Build Configuration:**
    *   Use Makefiles or other build tools to automate the build process.
    *   Define dependencies and build targets.
*   **Linting and Formatting:**
    *   Use Shellcheck for linting.
    *   Use a code formatter like `shfmt` to ensure consistent formatting.
*   **Deployment Best Practices:**
    *   Package your scripts with any necessary dependencies.
    *   Use a version control system like Git to track changes.
    *   Automate the deployment process using tools like Ansible or Puppet.
*   **CI/CD Integration:**
    *   Integrate your scripts with a CI/CD pipeline.
    *   Run tests and linters automatically.
    *   Deploy your scripts to production automatically.

## Additional Considerations

*   **Shebang:** Always start your scripts with a shebang line specifying the Zsh interpreter:
    zsh
    #!/usr/bin/env zsh
    
*   **Coding Style:** Follow a consistent coding style to improve readability.
*   **Documentation:** Document your scripts and functions using comments.
*   **Keep it Simple:** Aim for simplicity and clarity in your code.

By following these best practices, you can write Zsh scripts that are more reliable, maintainable, and secure.