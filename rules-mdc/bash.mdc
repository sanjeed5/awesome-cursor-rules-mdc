---
description: Enforces modern, robust, and maintainable bash scripting practices, focusing on error handling, quoting, variable management, and code organization.
globs: **/*
---
# bash Best Practices

This guide outlines the definitive best practices for writing bash scripts. Adhere to these rules to ensure your scripts are reliable, readable, and maintainable by the entire team.

## 1. Code Organization and Structure

### 1.1 Shebang and Interpreter Options
Always start scripts with `#!/usr/bin/env bash` for portability. Immediately follow with `set -euo pipefail` to ensure robust error handling.

*   `set -e` (errexit): Exit immediately if a command exits with a non-zero status.
*   `set -u` (nounset): Treat unset variables as an error and exit immediately.
*   `set -o pipefail`: The return value of a pipeline is the status of the last command to exit with a non-zero status, or zero if all commands exit successfully.

❌ BAD:
```bash
#!/bin/bash
# Missing error handling
```

✅ GOOD:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Script logic here
```

### 1.2 File Header and Comments
Every script must begin with a descriptive header. Use clear, concise comments for functions and complex logic.

```bash
#!/usr/bin/env bash
set -euo pipefail

## my-script.sh
#
# Description: This script performs a critical system backup.
# It backs up configuration files and database dumps to a remote server.
#
# Usage: ./my-script.sh [--full | --incremental]
#
# Globals:
#   BACKUP_DIR
#   REMOTE_HOST
#
# Arguments:
#   --full: Perform a full backup.
#   --incremental: Perform an incremental backup.
#
# Returns:
#   0 on success, non-zero on error.
#
# Author: Your Name <your.email@example.com>
# Date: 2025-01-01
#
###############################################################################

# ... rest of the script
```

### 1.3 Functions and `main` Entry Point
Encapsulate all script logic within functions. Use a `main` function as the primary entry point. This improves readability, reusability, and variable scoping.

❌ BAD:
```bash
#!/usr/bin/env bash
set -euo pipefail
# Script logic directly in the global scope
echo "Starting operation..."
ls -l /tmp
# ...
```

✅ GOOD:
```bash
#!/usr/bin/env bash
set -euo pipefail

# ... file header ...

# Performs a specific sub-task.
# Arguments:
#   $1 - The item to process.
process_item() {
  local item="${1}"
  echo "Processing: ${item}"
  # ...
}

# Main entry point for the script.
# Arguments:
#   $@ - All command-line arguments.
main() {
  echo "Script started."
  for arg in "$@"; do
    process_item "${arg}"
  done
  echo "Script finished."
}

# Call the main function with all script arguments.
main "$@"
```

### 1.4 Variable Scoping and Naming
Declare variables with `local` inside functions. Use `UPPER_SNAKE_CASE` for global `readonly` constants and `lower_snake_case` for local variables and function names.

❌ BAD:
```bash
#!/usr/bin/env bash
set -euo pipefail

GLOBAL_VAR="value" # Not readonly, could be modified
my_func() {
  temp_var="another_value" # Not local, pollutes global scope
}
```

✅ GOOD:
```bash
#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG_FILE="/etc/my_app/config.conf" # Global constant
readonly LOG_LEVEL="INFO"

my_function() {
  local temp_message="This is a local message."
  echo "${LOG_LEVEL}: ${temp_message}"
  # ...
}

main() {
  my_function
}

main "$@"
```

### 1.5 Indentation and Line Length
Use 2 spaces for indentation. Limit lines to approximately 80 characters. Break long lines with backslashes `\` for readability.

❌ BAD:
```bash
if [[ "${long_variable_name_one}" == "${long_variable_name_two}" && "${another_long_variable}" -gt 100 ]]; then echo "This line is way too long and hard to read."; fi
```

✅ GOOD:
```bash
if [[ "${long_variable_name_one}" == "${long_variable_name_two}" && \
      "${another_long_variable}" -gt 100 ]]; then
  echo "This line is readable."
fi
```

### 1.6 Cleanup with `trap`
Use `trap` to ensure resources (e.g., temporary files) are cleaned up even if the script exits unexpectedly.

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly TMP_DIR="$(mktemp -d)"

cleanup() {
  echo "Cleaning up temporary directory: ${TMP_DIR}" >&2
  rm -rf "${TMP_DIR}"
}

# Trap EXIT, INT (Ctrl+C), and TERM signals to run cleanup
trap cleanup EXIT INT TERM

main() {
  echo "Working in temporary directory: ${TMP_DIR}"
  touch "${TMP_DIR}/temp_file.txt"
  # ... script logic ...
  # No need to explicitly call cleanup, trap handles it.
}

main "$@"
```

## 2. Common Patterns and Anti-patterns

### 2.1 Always Quote Variables and Command Substitutions
This is the single most important rule. Always double-quote variable expansions (`"${var}"`) and use `$(command)` for command substitution to prevent word splitting and globbing.

❌ BAD:
```bash
files="file1.txt file with spaces.txt"
for f in $files; do # Word splitting occurs
  echo "Processing: $f" # Unquoted, will cause issues with spaces/globs
done

output=`ls -l` # Deprecated backticks
```

✅ GOOD:
```bash
files=("file1.txt" "file with spaces.txt") # Use arrays for lists
for f in "${files[@]}"; do # Correctly iterates over array elements
  echo "Processing: \"${f}\"" # Quoted, handles spaces correctly
done

output=$(ls -l) # Modern command substitution
```

### 2.2 Prefer `[[ ... ]]` for Conditionals
Use the bash-specific `[[ ... ]]` construct over `[ ... ]` (test). `[[ ... ]]` is safer, handles quoting internally, and supports advanced features like regex matching (`=~`).

❌ BAD:
```bash
if [ "$my_var" = "value" ]; then # Prone to issues if my_var is empty or contains spaces
  echo "Legacy test"
fi
```

✅ GOOD:
```bash
if [[ "${my_var}" == "value" ]]; then # Safer, quoting generally recommended for clarity
  echo "Modern test"
fi

# Regex matching
if [[ "${filename}" =~ \.log$ ]]; then
  echo "It's a log file."
fi
```

### 2.3 Use Long Options for Readability
Prefer long-form command options (`--recursive`) over short-form (`-r`) in scripts for improved readability.

❌ BAD:
```bash
rm -rf "${dir}"
```

✅ GOOD:
```bash
rm --recursive --force -- "${dir}" # -- to separate options from arguments
```

### 2.4 Redirect Errors to `STDERR`
Ensure all error messages and diagnostic output go to `STDERR` (`>&2`), reserving `STDOUT` for the intended program output.

```bash
log_error() {
  printf "[ERROR] %s: %s\n" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${*}" >&2
}

if ! some_command; then
  log_error "Failed to execute some_command."
  exit 1
fi
```

### 2.5 Heredocs for Multi-line Strings
Use heredocs for multi-line strings. Quote the tag (`<<'EOF'`) to prevent variable expansion and command substitution within the heredoc.

❌ BAD:
```bash
echo "Hello ${USER},"
echo "This is a multi-line message."
echo "Current date: $(date)"
```

✅ GOOD (Literal):
```bash
cat <<'EOF'
Hello ${USER},
This is a literal multi-line message.
Current date: $(date)
EOF
```

✅ GOOD (Interpolated):
```bash
cat <<EOF
Hello ${USER},
This is an interpolated multi-line message.
Current date: $(date)
EOF
```

### 2.6 Arithmetic with `(( ... ))`
Use `(( ... ))` for arithmetic operations. It's cleaner and safer than `expr` or `let`.

❌ BAD:
```bash
COUNT=`expr $COUNT + 1`
let COUNT=COUNT+1
```

✅ GOOD:
```bash
count=0
((count++))
echo "${count}" # Output: 1

num_a=10
num_b=5
result=$((num_a * num_b))
echo "${result}" # Output: 50
```

### 2.7 Scoped Directory Changes
When changing directories, use a subshell `(cd ...)` or `pushd`/`popd` to ensure the change is temporary and doesn't affect the rest of the script.

❌ BAD:
```bash
cd /tmp/my_app
# ... operations ...
cd - # Easy to forget, or fail if previous dir is gone
```

✅ GOOD:
```bash
(
  cd /tmp/my_app || exit 1 # Subshell, changes only apply here
  echo "Current directory in subshell: $(pwd)"
  # ... operations ...
)
echo "Current directory outside subshell: $(pwd)" # Original directory
```

## 3. Common Pitfalls and Gotchas

### 3.1 `while read` in a Pipe
Variables set inside a `while read` loop that is part of a pipeline will not persist outside the loop, as the loop runs in a subshell.

❌ BAD:
```bash
count=0
ls | while read -r file; do
  ((count++)) # count is modified in a subshell
done
echo "Total files: ${count}" # Will likely be 0
```

✅ GOOD:
```bash
count=0
# Use process substitution or redirect file directly
while read -r file; do
  ((count++))
done < <(ls) # Process substitution
echo "Total files: ${count}" # Correct count

# Alternative: mapfile (Bash 4+)
mapfile -t files_array < <(ls)
count="${#files_array[@]}"
echo "Total files: ${count}"
```

### 3.2 `sudo` with Redirection
Redirection (`>`) happens *before* `sudo` executes the command. To write to a root-owned file, run the entire command under `sudo`.

❌ BAD:
```bash
echo "Sensitive data" > /root/protected_file # Redirection by current user
```

✅ GOOD:
```bash
echo "Sensitive data" | sudo tee /root/protected_file > /dev/null
# OR
sudo bash -c 'echo "Sensitive data" > /root/protected_file'
```

### 3.3 Avoid `eval`
`eval` is a security risk and makes scripts hard to debug. Avoid it unless absolutely necessary and you fully control the input.

❌ BAD:
```bash
user_input="rm -rf /"
eval "${user_input}" # Dangerous!
```

✅ GOOD:
```bash
# Refactor to avoid eval. Use case statements, arrays, or functions instead.
# If dynamic command execution is truly needed, validate and sanitize input rigorously.
```

## 4. Testing Approaches

### 4.1 `shellcheck`
Integrate `shellcheck` into your workflow. It's an invaluable static analysis tool that catches common pitfalls and warns about bad practices.

```bash
# Run shellcheck on your script
shellcheck myscript.sh
```

### 4.2 Debugging with `set -x`
Use `set -x` (xtrace) for debugging. It prints each command and its arguments after expansion. Localize its use to specific sections.

```bash
#!/usr/bin/env bash
set -euo pipefail

my_debug_function() {
  set -x # Start tracing
  local debug_var="hello"
  echo "Debug var: ${debug_var}"
  set +x # Stop tracing
}

main() {
  echo "Before debug function."
  my_debug_function
  echo "After debug function."
}

main "$@"
```

### 4.3 Syntax Check with `bash -n`
Before running, perform a dry run to check for syntax errors.

```bash
bash -n myscript.sh
```