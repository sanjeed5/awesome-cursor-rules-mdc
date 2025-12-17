---
description: Definitive guidelines for building maintainable, performant, and type-safe Vue 3 applications using modern best practices, Composition API, and TypeScript.
globs: **/*.{js,jsx,ts,tsx,vue}
---
# Vue 3 Best Practices

This guide outlines the essential rules for developing high-quality Vue 3 applications. Adhere to these standards to ensure consistency, readability, and long-term maintainability across our codebase.

## 1. Foundational Setup: Linting & Formatting

**Always use ESLint and Prettier.** This combination enforces consistent code style and catches common errors automatically, streamlining development and code reviews. Integrate them into your IDE for real-time feedback and format-on-save.

## 2. Code Organization & Naming

### 2.1. Single-File Components (SFCs)
Each component **must** reside in its own `.vue` file.

❌ BAD
```js
// app.component('TodoList', { /* ... */ })
// app.component('TodoItem', { /* ... */ })
```

✅ GOOD
```
components/
├── TodoList.vue
└── TodoItem.vue
```

### 2.2. SFC Filename Casing
Use **PascalCase** for all SFC filenames. This aligns with component referencing in templates and JS(X) and improves editor autocompletion.

❌ BAD
```
components/
├── my-component.vue
└── myComponent.vue
```