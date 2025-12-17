---
description: This guide provides definitive, opinionated best practices for writing clean, performant, and maintainable Vue 3 applications using modern patterns like the Composition API and Pinia.
globs: **/*.{vue,js,ts}
---
# Vue Best Practices

This document outlines the definitive best practices for Vue 3 development. Adhere to these guidelines to ensure consistent, high-quality, and performant code across our projects.

## Code Organization and Structure

### 1. Single-File Components (SFCs) Per File

Always place each Vue component in its own `.vue` file. This improves discoverability, modularity, and version control.

❌ BAD
```vue
<!-- components/TodoList.vue -->
<template>...</template>
<script>
export default {
  // ... TodoList logic
}
</script>

<!-- components/TodoItem.vue (in the same file) -->
<template>...</template>
<script>
export default {
  // ... TodoItem logic
}
</script>
```

✅ GOOD
```vue
<!-- components/TodoList.vue -->
<template>...</template>
<script setup>
// ... TodoList logic
</script>

<!-- components/TodoItem.vue -->
<template>...</template>
<script setup>
// ... TodoItem logic
</script>
```

### 2. Consistent Component Naming

Use **PascalCase** for Single-File Component filenames (e.g., `MyComponent.vue`). This aligns with how components are referenced in templates and JavaScript, improving IDE autocompletion.

❌ BAD
```
components/
├── my-component.vue
└── Mycomponent.vue
```

✅ GOOD
```
components/
├── MyComponent.vue
└── AnotherComponent.vue
```

### 3. Base Component Prefixing

Prefix "base" or "presentational" components (those that only contain HTML, other base components, or 3rd-party UI components, and no global state) with `Base`, `App`, or `V`.

❌ BAD
```
components/
├── Button.vue
└── Icon.vue
```

✅ GOOD
```
components/
├── BaseButton.vue
└── BaseIcon.vue
```

### 4. Tightly Coupled Component Naming

Child components tightly coupled to a single parent must include the parent's name as a prefix. This groups related files alphabetically and clarifies relationships.

❌ BAD
```
components/
├── UserProfile.vue
├── Avatar.vue
└── EditButton.vue
```

✅ GOOD
```
components/
├── UserProfile.vue
├── UserProfileAvatar.vue
└── UserProfileEditButton.vue
```

## Common Patterns and Anti-patterns

### 5. Always Use `key` with `v-for`

Provide a unique `key` attribute on `v-for` loops. This gives Vue a stable reference for each item, crucial for efficient DOM updates, transitions, and component state management.

❌ BAD
```vue
<template>
  <div v-for="item in items">{{ item.name }}</div>
</template>
```

✅ GOOD
```vue
<template>
  <div v-for="item in items" :key="item.id">{{ item.name }}</div>
</template>
```

### 6. Kebab-Case for Custom Events

Emit and listen to custom events using `kebab-case`. This is consistent with HTML event naming conventions.

❌ BAD
```vue
<!-- ChildComponent.vue -->
<script setup>
const emit = defineEmits(['closeWindow']);
emit('closeWindow');
</script>

<!-- ParentComponent.vue -->
<template>
  <ChildComponent @closeWindow="handleClose" />
</template>
```

✅ GOOD
```vue
<!-- ChildComponent.vue -->
<script setup>
const emit = defineEmits(['close-window']);
emit('close-window');
</script>

<!-- ParentComponent.vue -->
<template>
  <ChildComponent @close-window="handleClose" />
</template>
```

### 7. Prop Naming Consistency

Declare props in `camelCase` within `<script setup>`, but use `kebab-case` when passing them in templates. Vue automatically handles the conversion.

❌ BAD
```vue
<!-- ChildComponent.vue -->
<script setup>
defineProps({
  titletext: String,
});
</script>

<!-- ParentComponent.vue -->
<template>
  <ChildComponent titletext="Hello" />
</template>
```

✅ GOOD
```vue
<!-- ChildComponent.vue -->
<script setup>
defineProps({
  titleText: String, // camelCase in script
});
</script>

<!-- ParentComponent.vue -->
<template>
  <ChildComponent title-text="Hello" /> <!-- kebab-case in template -->
</template>
```

### 8. Validate Props Explicitly

Always define prop types, `required` flags, and custom validators. This prevents runtime errors, improves component API clarity, and acts as self-documentation.

❌ BAD
```vue
<script setup>
defineProps(['status', 'count']);
</script>
```

✅ GOOD
```vue
<script setup>
defineProps({
  status: {
    type: String,
    required: true,
    validator: (value) => ['syncing', 'synced', 'error'].includes(value),
  },
  count: {
    type: Number,
    default: 0,
  },
});
</script>
```

### 9. Avoid `v-if` on `v-for` Elements

Never use `v-if` directly on an element with `v-for`. `v-for` takes precedence, leading to inefficient iteration over the entire list before filtering. Filter the list in a `computed` property instead.

❌ BAD
```vue
<template>
  <div v-for="product in products" v-if="product.price < 500">
    {{ product.name }}
  </div>
</template>
```

✅ GOOD
```vue
<script setup>
import { computed, ref } from 'vue';

const products = ref([/* ... */]);
const cheapProducts = computed(() => {
  return products.value.filter((product) => product.price < 500);
});
</script>

<template>
  <div v-for="product in cheapProducts" :key="product.id">
    {{ product.name }}
  </div>
</template>
```

## Performance Considerations

### 10. Pure and Deterministic Computed Properties

Write computed properties that are pure and deterministic (given the same inputs, they always produce the same output). This allows Vue's reactivity engine to cache results aggressively, boosting performance. Avoid side effects within computed properties.

❌ BAD
```vue
<script setup>
import { computed, ref } from 'vue';
const items = ref([1, 2, 3]);
const expensiveComputation = computed(() => {
  // Side effect: Modifies items, making it non-deterministic
  items.value.push(Math.random());
  return items.value.reduce((sum, val) => sum + val, 0);
});
</script>
```

✅ GOOD
```vue
<script setup>
import { computed, ref } from 'vue';
const items = ref([1, 2, 3]);
const sumOfItems = computed(() => {
  // Pure: Only depends on items.value, no side effects
  return items.value.reduce((sum, val) => sum + val, 0);
});
</script>
```

## State Management

### 11. Use Pinia for Global State

For global or complex shared state, use Pinia. It's the official and recommended state management library for Vue 3, offering a simpler API, better TypeScript support, and smaller bundle sizes than Vuex.

❌ BAD
```vue
<!-- Using Vuex (older pattern) -->
<script setup>
import { useStore } from 'vuex';
const store = useStore();
const user = computed(() => store.state.user);
</script>
```

✅ GOOD
```vue
<!-- Using Pinia -->
<script setup>
import { useUserStore } from '@/stores/user'; // Assuming @/stores/user.js
import { storeToRefs } from 'pinia';

const userStore = useUserStore();
const { user } = storeToRefs(userStore); // Keep reactivity for state properties
</script>
```