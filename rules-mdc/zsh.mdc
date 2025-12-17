---
description: This guide defines the definitive best practices for writing Zsh scripts and configuration files, ensuring consistency, robustness, and maintainability across our projects.
globs: **/*
---
# zsh Best Practices

Zsh is our default shell for interactive use and scripting. This guide outlines mandatory practices to ensure all Zsh code is robust, readable, and performant.

## 1. Core Principles

### 1.1 Interpreter & Strict Mode
Always specify `zsh` as the interpreter and enable strict mode for early error detection.

✅ **GOOD:**
```zsh
#!/usr/bin/env zsh

# Enable strict mode:
# errreturn: Exit immediately if a command exits with a non-zero status.
# nounset: Treat unset variables as an error.
# pipefail: The exit status of a pipeline is the status of the last command to exit with a non-zero status, or zero if all commands exit successfully.
setopt errreturn nounset pipefail
```

❌ **BAD:**
```zsh
#!/bin/bash # Wrong interpreter
# No strict mode options
```

### 1.2 ShellCheck Integration
ShellCheck is non-negotiable. Integrate it into your editor and CI pipeline.

✅ **GOOD:**
```zsh
# .github/workflows/ci.yml
name: ShellCheck

on: [push, pull_request]

jobs:
  shellcheck:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run ShellCheck
      uses: ludeeus/action-shellcheck@v2
      with:
        scandir: './' # Scan all shell scripts in the repository
```

## 2. Code Organization and Structure

### 2.1 File Header & Comments
Every script must start with a descriptive header. Functions require clear comments.

✅ **GOOD:**
```zsh
#!/usr/bin/env zsh
#
# Description: Manages project dependencies via a custom tool.
#
# setopt errreturn nounset pipefail

#######################################
# Fetches and installs project dependencies.
# Globals:
#   PROJECT_ROOT
# Arguments:
#   None
# Outputs:
#   Writes status messages to stderr.
# Returns:
#   0 if dependencies are installed, non-zero on error.
#######################################
function install_dependencies() {
  # ... implementation ...
}
```

### 2.2 Function Location & `main`
Define all functions before their first use. Use a `main` function for the script's entry point.

✅ **GOOD:**
```zsh
#!/usr/bin/env zsh
setopt errreturn nounset pipefail

function _log_info() {
  echo "[INFO] $*" >&2
}

function _process_data() {
  _log_info "Processing data for: $1"
  # ...
}

function main() {
  _log_info "Starting data processing..."
  _process_data "report_A"
  _process_data "report_B"
  _log_info "Processing complete."
}

main "$@"
```

## 3. Formatting

### 3.1 Indentation & Line Length
Indent with 2 spaces. Limit lines to 80 characters.

✅ **GOOD:**
```zsh
if [[ -n "${VAR}" ]]; then
  echo "Variable is set."
fi

long_command_name --option-one value \
  --option-two another-value \
  --final-option
```

❌ **BAD:**
```zsh
if [[ -n "${VAR}" ]]; then
    echo "Variable is set."
fi # 4 spaces

long_command_name --option-one value --option-two another-value --final-option # Too long
```

### 3.2 Quoting & Variable Expansion
Always quote variables and command substitutions unless you explicitly need word splitting. Prefer `${parameter}`.

✅ **GOOD:**
```zsh
local my_file="report.txt"
local content="$(cat "${my_file}")" # Command substitution quoted
echo "Processing file: ${my_file}"  # Variable quoted, preferred syntax
```

❌ **BAD:**
```zsh
local my_file=report.txt
local content=$(cat $my_file) # Unquoted, prone to word splitting/globbing
echo "Processing file: $my_file" # Legacy syntax
```

## 4. Naming Conventions

### 4.1 Functions & Variables
- **Functions:** `lower_snake_case`. Prefix internal functions with `_`.
- **Constants/Environment Variables:** `ALL_CAPS`.
- **Local Variables:** `local lower_snake_case`.

✅ **GOOD:**
```zsh
local MY_CONSTANT="fixed_value"
local _internal_helper_var="temp"

function calculate_total() {
  local item_count="${1}"
  local unit_price="${2}"
  echo $(( item_count * unit_price ))
}
```

❌ **BAD:**
```zsh
local myConstant="value" # CamelCase
function CalculateTotal() { # PascalCase
  local ItemCount="${1}" # PascalCase
}
```

## 5. Zsh-Specific Patterns

### 5.1 Zsh Arrays
Use Zsh's native array handling. `${array}` expands to all elements. Use `"${(@)array}"` to preserve empty entries.

✅ **GOOD:**
```zsh
local my_array=("apple" "" "banana")
echo "All elements: ${my_array}"        # apple banana (empty removed)
echo "Preserving empty: ${(@)my_array}" # apple  banana (empty preserved)

# Iterate safely
for item in "${(@)my_array}"; do
  echo "Item: '${item}'"
done
```

❌ **BAD:**
```zsh
local my_array=("apple" "" "banana")
echo "Bash style: ${my_array[@]}" # Works, but not idiomatic Zsh
```

### 5.2 Extended Globbing
Enable `extended_glob` for powerful pattern matching.

✅ **GOOD:**
```zsh
setopt extended_glob

# Find files not ending with .txt or .log
ls -d *~*.(txt|log)

# Case-insensitive glob
ls -d (#i)*.txt
```

### 5.3 Parameter Expansion for Path Manipulation
Avoid `dirname` and `basename` external commands.

✅ **GOOD:**
```zsh
local full_path="/path/to/my/file.txt"
local dir_name="${full_path:h}"  # /path/to/my
local base_name="${full_path:t}" # file.txt
local absolute_path="${full_path:A}" # Resolve symlinks and get absolute path
```

❌ **BAD:**
```zsh
local full_path="/path/to/my/file.txt"
local dir_name="$(dirname "${full_path}")"
local base_name="$(basename "${full_path}")"
```

### 5.4 Zsh-Native Filtering (Skipping `grep`/`tr`)
Leverage Zsh's parameter expansion for filtering and transformations.

✅ **GOOD:**
```zsh
# Filter array (like grep)
local lines=("line one" "another line" "third line")
local matched_lines=(${(M)lines:#*line*}) # Matches lines containing "line"
print -l "${matched_lines[@]}"

# Transform string (like tr)
local text="hello world"
local transformed_text="${text//[aeiou]/_}" # h_ll_ w_rld
```

❌ **BAD:**
```zsh
local lines=("line one" "another line" "third line")
local matched_lines="$(printf "%s\n" "${lines[@]}" | grep "line")" # External command, subshell
```

### 5.5 Ternary Expressions
Use `:+` and `:-` for concise conditional assignments.

✅ **GOOD:**
```zsh
local debug_mode="true"
local log_level="${debug_mode:+DEBUG}:INFO" # If debug_mode is set, prepend DEBUG
echo "Log level: ${log_level}" # Output: Log level: DEBUG:INFO

local debug_mode=""
local log_level="${debug_mode:+DEBUG}:INFO"
echo "Log level: ${log_level}" # Output: Log level: :INFO
```

## 6. Anti-Patterns & Pitfalls

### 6.1 Avoid `eval`
`eval` is a security risk and makes code hard to debug. Never use it.

❌ **BAD:**
```zsh
local cmd="ls -l"
eval "${cmd}" # Dangerous!
```

### 6.2 `source` with Caution
Only `source` trusted files. For external tools, prefer explicit execution or `eval "$(tool init zsh)"` if the tool is trusted and designed for it.

✅ **GOOD:**
```zsh
# .zshrc
# Source trusted local configuration
source "${ZDOTDIR:-$HOME}/.zsh_aliases"

# For trusted tools that provide init scripts
eval "$(zoxide init zsh)"
```

❌ **BAD:**
```zsh
source "/tmp/untrusted_script.zsh"
```

### 6.3 Minimal `.zshrc`
Keep your `.zshrc` lean. Avoid heavy frameworks like Oh My Zsh unless absolutely necessary. Prefer lightweight tools and manual configuration.

✅ **GOOD:**
```zsh
# .zshrc
HISTFILE=~/.zsh_history
HISTSIZE=100000
SAVEHIST=100000
setopt HIST_SAVE_NO_DUPS INC_APPEND_HISTORY AUTO_PUSHD PUSHD_IGNORE_DUPS PUSHD_SILENT autocd
autoload -U compinit; compinit

# Prompt via Starship (external, lightweight)
eval "$(starship init zsh)"
```

❌ **BAD:**
```zsh
# .zshrc
# Too many plugins, slow startup
source $ZSH/oh-my-zsh.sh
plugins=(git docker web-search history-substring-search ...)
```