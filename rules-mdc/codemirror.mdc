---
description: This guide defines best practices for building and maintaining CodeMirror 6-based editors, focusing on modularity, immutable state, extension-driven configuration, and modern JavaScript/TypeScript patterns.
globs: **/*.{js,jsx}
---
# CodeMirror 6 Best Practices

This document outlines the definitive best practices for developing with CodeMirror. We exclusively target **CodeMirror 6 (CM6)**. Any reference to CodeMirror 5 (CM5) is for historical context only; **CM5 must not be used in new development and should be actively migrated away from.**

## Code Organization and Structure

### 1. Always Use CodeMirror 6

CM6 is a complete rewrite with a modern, modular architecture. It offers superior performance, extensibility, and maintainability. CM5 is deprecated and unsupported.

❌ **BAD: Using CodeMirror 5**

```javascript
// Old CM5 global API
var myCodeMirror = CodeMirror(document.body, {
  value: "function myScript(){}",
  mode: "javascript"
});
```

✅ **GOOD: Using CodeMirror 6**

```javascript
import { EditorState } from "@codemirror/state";
import { EditorView, basicSetup } from "codemirror";
import { javascript } from "@codemirror/lang-javascript";

const view = new EditorView({
  state: EditorState.create({
    doc: "console.log('Hello CM6!');",
    extensions: [basicSetup, javascript()],
  }),
  parent: document.body,
});
```

### 2. Import Only What You Need (Modular Imports)

CM6 is designed for tree-shaking. Only import the specific packages and modules required to minimize bundle size and improve load times.

❌ **BAD: Importing monolithic packages (if they existed for CM6)**

```javascript
// Hypothetical bad example: importing a giant 'codemirror-all' package
import { EditorView, EditorState, basicSetup, javascript, ... } from "codemirror-all";
```

✅ **GOOD: Granular, specific imports**

```javascript
import { EditorState } from "@codemirror/state"; // Core state management
import { EditorView, keymap } from "@codemirror/view"; // Core view rendering
import { defaultKeymap } from "@codemirror/commands"; // Standard editing commands
import { javascript } from "@codemirror/lang-javascript"; // Language support
```

### 3. Configure Exclusively via Extensions

All editor features, from basic setup to complex language services, must be added as extensions to the `EditorState`. Avoid direct DOM manipulation or non-extension-based configuration.

❌ **BAD: Direct property assignment or DOM manipulation**

```javascript
const view = new EditorView({ state: EditorState.create({ doc: "" }) });
// Don't do this: Bypassing the extension system
view.dom.style.fontSize = "16px";
```

✅ **GOOD: Everything is an extension**

```javascript
import { EditorState } from "@codemirror/state";
import { EditorView, lineNumbers } from "@codemirror/view";
import { oneDark } from "@codemirror/theme-one-dark"; // A theme extension

const view = new EditorView({
  state: EditorState.create({
    doc: "const x = 1;",
    extensions: [
      lineNumbers(), // Line numbers extension
      oneDark, // Theme extension
      EditorView.lineWrapping, // Line wrapping extension
      EditorState.tabSize.of(2), // Configuration option as an extension
    ],
  }),
  parent: document.body,
});
```

## Common Patterns and Anti-patterns

### 4. Embrace Immutable State Management

`EditorState` objects are immutable. All changes must be applied by creating a `Transaction` and dispatching it to the `EditorView`. This enables reliable undo/redo, time-travel debugging, and predictable behavior.

❌ **BAD: Mutating `EditorState` or `Text` directly**

```javascript
let state = EditorState.create({ doc: "hello" });
// This will NOT work as expected and breaks CM6's core design
// state.doc = Text.of("world");
// console.log(state.doc.toString()); // Still "hello"
```

✅ **GOOD: Using `Transaction` and `dispatch`**

```javascript
import { EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";

const startState = EditorState.create({ doc: "Hello World" });
const view = new EditorView({ state: startState, parent: document.body });

// Create a transaction to change the document
const transaction = view.state.update({
  changes: { from: 0, to: 5, insert: "Goodbye" }, // Change "Hello" to "Goodbye"
  selection: { anchor: 7 }, // Move cursor
  // Add other state changes like effects or annotations here
});

view.dispatch(transaction); // Apply the transaction
console.log(view.state.doc.toString()); // "Goodbye World"
```

### 5. Prioritize TypeScript for New Code

For new CodeMirror components, always use TypeScript. It provides static type checking, improved IDE support, and reduces runtime errors, leading to more robust and maintainable code.

❌ **BAD: Writing new complex modules in plain JavaScript**

```javascript
// my-complex-cm-plugin.js
export function myComplexPlugin() {
  // No type safety, easy to introduce bugs
  return ViewPlugin.define(view => ({
    update(update) {
      if (update.docChanged) {
        // What type is update.docChanged?
      }
    }
  }));
}
```

✅ **GOOD: Writing new complex modules in TypeScript**

```typescript
// my-complex-cm-plugin.ts
import { EditorView, ViewPlugin, ViewUpdate } from "@codemirror/view";
import { Extension } from "@codemirror/state";

export function myComplexPlugin(): Extension {
  return ViewPlugin.define(view => ({
    // Explicitly typed parameters for clarity and safety
    update(update: ViewUpdate) {
      if (update.docChanged) {
        console.log("Document changed:", update.changes.toString());
      }
    },
  }));
}
```

## Performance Considerations

### 6. Batch Editor Updates

Avoid dispatching multiple transactions in quick succession. Group related changes into a single `Transaction` to minimize DOM updates and improve performance.

❌ **BAD: Frequent, unbatched dispatches**

```javascript
// In a loop or event handler
for (let i = 0; i < 100; i++) {
  view.dispatch(view.state.update({ changes: { from: i, insert: "a" } }));
}
```

✅ **GOOD: Batching changes into a single transaction**

```javascript
let changes = [];
for (let i = 0; i < 100; i++) {
  changes.push({ from: i, insert: "a" });
}
view.dispatch(view.state.update({ changes })); // Single dispatch
```

### 7. Debounce/Throttle External Event Handlers

When external events (e.g., window resize, network responses, user input outside the editor) trigger editor updates or expensive computations, always debounce or throttle them to prevent excessive processing.

```javascript
import { EditorView } from "@codemirror/view";
import { EditorState } from "@codemirror/state";
import { javascript } from "@codemirror/lang-javascript";

const view = new EditorView({
  state: EditorState.create({
    doc: "function expensiveLint() { /* ... */ }",
    extensions: [javascript()],
  }),
  parent: document.body,
});

// Assume `lintDocument` is an expensive operation
const debouncedLint = debounce(() => {
  // Trigger linting based on current editor state
  console.log("Linting document...");
}, 500);

// Listen for editor state changes
view.dom.addEventListener("input", debouncedLint);

function debounce(func, delay) {
  let timeout;
  return function(...args) {
    const context = this;
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(context, args), delay);
  };
}
```

## Common Pitfalls and Gotchas

### 8. Never Mix CodeMirror 5 and 6

The architectures are fundamentally different. Attempting to use CM5 add-ons or APIs with a CM6 editor will lead to unpredictable behavior and errors.

❌ **BAD: Using CM5 add-ons with CM6**

```javascript
// This will NOT work with CM6
import "codemirror/addon/search/searchcursor";
```

✅ **GOOD: Use CM6-native extensions**

```javascript
// For search, use CM6's @codemirror/search
import { search, searchKeymap, highlightSelectionMatches } from "@codemirror/search";
import { EditorState } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";

const view = new EditorView({
  state: EditorState.create({
    doc: "search me",
    extensions: [
      search(),
      keymap.of(searchKeymap),
      highlightSelectionMatches(),
    ],
  }),
  parent: document.body,
});
```

### 9. Understand Extension Precedence

When multiple extensions affect the same aspect of the editor (e.g., keymaps, decorations), their order in the `extensions` array and explicit `Prec` modifiers determine precedence. Debug carefully if an extension isn't behaving as expected.

```javascript
import { EditorState, Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { defaultKeymap } from "@codemirror/commands";

// A custom keymap that should override default behavior
const myCustomKeymap = keymap.of([
  {
    key: "Mod-s",
    run: (view) => {
      console.log("Custom save action!");
      return true; // Indicate that the event was handled
    },
  },
]);

const view = new EditorView({
  state: EditorState.create({
    doc: "Press Cmd-S",
    extensions: [
      defaultKeymap, // Default keymap is defined first
      Prec.highest(myCustomKeymap), // Custom keymap with highest precedence will override
    ],
  }),
  parent: document.body,
});
```

## Testing Approaches

### 10. Unit Test `EditorState` Logic

Isolate and unit test functions that operate on or produce `EditorState` objects. Since `EditorState` is immutable, these functions are pure and easy to test without a browser environment.

```javascript
// my-state-transformer.js
import { EditorState } from "@codemirror/state";

export function addCommentToLine(state, lineNumber, commentText) {
  const line = state.doc.line(lineNumber);
  return state.update({
    changes: { from: line.from, insert: `// ${commentText}\n` + line.text },
  });
}

// my-state-transformer.test.js
import { EditorState } from "@codemirror/state";
import { addCommentToLine } from "./my-state-transformer";

test("addCommentToLine adds comment correctly", () => {
  const initialState = EditorState.create({ doc: "line 1\nline 2" });
  const newState = addCommentToLine(initialState, 2, "My comment");
  expect(newState.doc.toString()).toBe("line 1\n// My comment\nline 2");
});
```

### 11. Use End-to-End Tests for `EditorView` Interactions

For testing UI interactions (typing, selection, scrolling, extension rendering), use browser-based end-to-end testing frameworks like Playwright or Cypress. These simulate real user behavior.

```javascript
// playwright.test.js (conceptual example)
import { test, expect } from "@playwright/test";

test("editor displays line numbers and allows typing", async ({ page }) => {
  await page.goto("http://localhost:3000/editor"); // Your app's editor page

  const editor = page.locator(".cm-editor");
  await expect(editor).toBeVisible();

  // Check if line numbers are visible
  await expect(page.locator(".cm-lineNumbers")).toBeVisible();

  // Type into the editor
  await editor.type("Hello, CodeMirror!");
  await expect(editor).toHaveText(/Hello, CodeMirror!/);
});
```