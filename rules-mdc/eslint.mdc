---
description: This guide defines the definitive ESLint configuration and best practices for our team, ensuring consistent code quality, early error detection, and seamless integration with modern JavaScript and TypeScript workflows.
globs: **/*.{js,jsx,ts,tsx,mts,cts}
---
# eslint Best Practices

ESLint is the bedrock of our JavaScript and TypeScript code quality. This guide outlines our definitive, opinionated approach to configuring and using ESLint, focusing on modern best practices for December 2025.

## 1. Core Configuration: Flat Config is Mandatory

Always use the new flat configuration format (`eslint.config.js`, `.mjs`, `.cjs`, or `.ts`). The legacy `.eslintrc` format is deprecated and **must not** be used.

### 1.1. Base Configuration

Start with the recommended and strict presets for robust error prevention and stylistic consistency.

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import { defineConfig } from 'eslint/config';

export default defineConfig([
  eslint.configs.recommended, // Essential base rules
  eslint.configs.strict,     // Stricter error-prevention rules
  eslint.configs.stylistic,  // Stylistic rules (handled by Prettier, see 1.3)
  // ... other configs and rules
]);
```

### 1.2. TypeScript Integration (Type-Aware Linting)

For TypeScript projects, enable type-aware linting. This provides powerful, deep analysis that catches subtle type-related issues.

```javascript
// eslint.config.ts (requires `jiti` or `ts-node` for execution)
import eslint from '@eslint/js';
import { defineConfig } from 'eslint/config';
import tseslint from 'typescript-eslint';

export default defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommendedTypeChecked, // TypeScript recommended rules with type-checking
  tseslint.configs.strictTypeChecked,      // Stricter TypeScript rules with type-checking
  tseslint.configs.stylisticTypeChecked,   // TypeScript stylistic rules (handled by Prettier)
  {
    languageOptions: {
      parser: tseslint.parser,
      parserOptions: {
        projectService: true, // Enable TypeScript's project service for type info
        tsconfigRootDir: import.meta.dirname, // Point to your project root
      },
    },
    files: ['**/*.ts', '**/*.tsx', '**/*.mts', '**/*.cts'], // Apply to TS files
  },
  // ... other configs and rules
]);
```

**Performance Note**: Type-aware linting adds overhead. This is a worthwhile trade-off for the increased safety and quality. Rely on IDE extensions for instant feedback and run full linting in CI.

### 1.3. Prettier Integration

ESLint handles code quality; Prettier handles code formatting. Use `eslint-config-prettier` to disable any ESLint rules that conflict with Prettier, preventing unnecessary warnings.

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import { defineConfig } from 'eslint/config';
import prettierConfig from 'eslint-config-prettier'; // Make sure this is last!

export default defineConfig([
  eslint.configs.recommended,
  // ... other configs (e.g., tseslint configs)
  prettierConfig, // Always place this last to override conflicting rules
  {
    rules: {
      // Custom rules or overrides go here
      'no-console': 'warn', // Example: warn on console logs
    },
  },
]);
```

## 2. Code Organization and Structure

Apply rules precisely where they're needed using `files` and `ignores`.

```javascript
// eslint.config.js
import { defineConfig } from 'eslint/config';

export default defineConfig([
  // ... base configs
  {
    files: ['src/**/*.js', 'src/**/*.jsx'], // Apply to JS/JSX files in src
    rules: {
      'react/jsx-uses-react': 'off', // Example: React 17+ doesn't need this
      'react/react-in-jsx-scope': 'off',
    },
  },
  {
    files: ['**/*.test.js', '**/*.spec.ts'], // Apply to test files
    rules: {
      'no-unused-expressions': 'off', // Allow chai-style assertions
      'jest/expect-expect': 'error',   // Ensure tests have assertions
    },
  },
  {
    ignores: ['dist/', 'node_modules/', 'coverage/'], // Globally ignore these directories
  },
]);
```

## 3. Common Patterns and Anti-patterns

Enforce these critical rules to maintain high code quality.

### 3.1. Immutability (`prefer-const`)

Always use `const` unless a variable's value is reassigned.

❌ BAD
```javascript
let name = 'Alice';
name = 'Bob'; // Reassigned, but `let` was used initially
```

✅ GOOD
```javascript
const name = 'Alice'; // Never reassigned
let age = 30;
age++; // Reassigned, `let` is appropriate
```

### 3.2. Unused Variables (`no-unused-vars`)

Remove dead code. Unused variables, functions, or imports indicate cruft.

❌ BAD
```javascript
const unusedVar = 10;
function doSomething() { /* ... */ } // Never called
```

✅ GOOD
```javascript
const usedVar = 10;
console.log(usedVar);
```

### 3.3. Consistent Returns (`consistent-return`)

Ensure functions either always return a value or never return one explicitly. Avoid implicit `undefined` returns.

❌ BAD
```javascript
function process(value) {
  if (value > 0) {
    return value * 2;
  }
  // Implicitly returns undefined here
}
```

✅ GOOD
```javascript
function process(value) {
  if (value > 0) {
    return value * 2;
  }
  return 0; // Explicit return
}

function logValue(value) {
  console.log(value); // No return value expected
}
```

### 3.4. No Else Return (`no-else-return`)

Simplify conditional logic by returning early.

❌ BAD
```javascript
function getValue(condition) {
  if (condition) {
    return 'A';
  } else {
    return 'B';
  }
}
```

✅ GOOD
```javascript
function getValue(condition) {
  if (condition) {
    return 'A';
  }
  return 'B'; // Simplified
}
```

### 3.5. Strict Equality (`eqeqeq`)

Always use `===` and `!==` to prevent type coercion issues.

❌ BAD
```javascript
if (value == null) { /* ... */ } // Coercion
```

✅ GOOD
```javascript
if (value === null || value === undefined) { /* ... */ } // Explicit check
if (value === 0) { /* ... */ } // No coercion
```

### 3.6. No Console Logs (`no-console`)

Prevent accidental `console.log` statements from reaching production.

```javascript
// eslint.config.js
export default defineConfig([
  // ...
  {
    rules: {
      'no-console': ['error', { allow: ['warn', 'error'] }], // Allow warn/error, but error on log
    },
  },
]);
```

### 3.7. No Magic Numbers (`no-magic-numbers`)

Replace unexplained numeric literals with named constants for readability and maintainability.

❌ BAD
```javascript
function calculateArea(radius) {
  return 3.14159 * radius * radius;
}
```

✅ GOOD
```javascript
const PI = 3.14159;
function calculateArea(radius) {
  return PI * radius * radius;
}
```

## 4. Common Pitfalls and Gotchas

### 4.1. Avoid `eslint-disable`

**Never** use `eslint-disable` comments. They mask underlying issues, accumulate technical debt, and compromise code quality. If a rule is genuinely problematic for a specific, rare case, discuss it with the team to adjust the global configuration or create a targeted override.

❌ BAD
```javascript
// eslint-disable-next-line no-console
console.log('Debug info');
```

✅ GOOD
```javascript
// Remove or refactor the problematic code.
// Or, if truly necessary, configure the rule globally to allow specific cases.
```

### 4.2. Configuration Order Matters

Ensure `eslint-config-prettier` is always the last configuration in your array to correctly disable conflicting rules.

```javascript
// eslint.config.js
import eslint from '@eslint/js';
import prettierConfig from 'eslint-config-prettier';

export default defineConfig([
  eslint.configs.recommended,
  // ... other plugins/configs like tseslint
  prettierConfig, // MUST BE LAST
  {
    rules: {
      // Your custom rules go here and will override previous configs if needed
    },
  },
]);
```

## 5. Testing Approaches

Integrate ESLint into your development workflow for proactive issue detection.

### 5.1. Pre-commit Hooks (Husky + lint-staged)

Enforce linting on staged files before every commit. This ensures only clean code enters the repository.

```json
// package.json
{
  "name": "your-project",
  // ...
  "devDependencies": {
    "husky": "^9.0.0",
    "lint-staged": "^15.0.0",
    "eslint": "^9.0.0"
  },
  "scripts": {
    "prepare": "husky",
    "lint": "eslint .",
    "lint:fix": "eslint --fix ."
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx,mts,cts}": "eslint --fix"
  }
}
```

Then, set up Husky:

```bash
npx husky init
npx husky add .husky/pre-commit "npx lint-staged"
```

### 5.2. CI/CD Integration

Run ESLint as a mandatory step in your Continuous Integration pipeline. This acts as a final gatekeeper for code quality.

```yaml
# .github/workflows/ci.yml (Example for GitHub Actions)
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint # Run the lint script defined in package.json
```