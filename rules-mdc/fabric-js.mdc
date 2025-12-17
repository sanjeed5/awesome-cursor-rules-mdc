---
description: This guide outlines definitive best practices for using fabric-js (v6.9.0+) in JavaScript/TypeScript projects, focusing on performance, maintainability, and modern API usage.
globs: **/*.{js,jsx}
---
# fabric-js Best Practices

This document provides definitive guidelines for developing with `fabric-js` (v6.9.0+), emphasizing performance, maintainability, and modern API adoption. Adhere to these rules to ensure robust and efficient canvas applications.

## 1. Code Organization and Structure

### 1.1. Embrace TypeScript and ES Modules

Always use TypeScript for type safety and leverage ES6 modules for efficient bundling. Import only the specific `fabric` components you need.

❌ BAD: Global `fabric` access, full library import
```javascript
// main.js
import 'fabric'; // Imports everything
const canvas = new fabric.Canvas('myCanvas');
canvas.add(new fabric.Rect({ left: 10, top: 10, width: 50, height: 50 }));
```

✅ GOOD: Modular, typed imports
```typescript
// main.ts
import { Canvas, Rect, IText } from 'fabric';

const canvas = new Canvas('myCanvas');
const rect = new Rect({ left: 10, top: 10, width: 50, height: 50, fill: 'red' });
canvas.add(rect);
```

### 1.2. Choose the Right Canvas Type

Use `StaticCanvas` for read-only displays or high-frame-rate scenes where interactivity is not required. It offers a significant performance boost by skipping event handling.

❌ BAD: Using `Canvas` when no interaction is needed
```typescript
// For displaying a static image or animation without user interaction
const canvas = new Canvas('myStaticCanvas');
canvas.add(someFabricObject);
canvas.renderAll();
```

✅ GOOD: Using `StaticCanvas` for static or high-performance rendering
```typescript
// For displaying a static image or animation without user interaction
import { StaticCanvas, Rect } from 'fabric';

const staticCanvas = new StaticCanvas('myStaticCanvas');
const rect = new Rect({ left: 10, top: 10, width: 50, height: 50, fill: 'blue' });
staticCanvas.add(rect);
staticCanvas.renderAll();
```

## 2. Performance Considerations

### 2.1. Disable Unnecessary Interactivity

Minimize overhead by disabling selection, controls, and borders on objects or the canvas itself when not needed.

❌ BAD: Default interactive objects everywhere
```typescript
const rect = new Rect({ left: 10, top: 10, width: 50, height: 50, fill: 'green' });
canvas.add(rect); // Selectable, has controls/borders by default
```

✅ GOOD: Optimize object interactivity
```typescript
const rect = new Rect({
  left: 10, top: 10, width: 50, height: 50, fill: 'green',
  selectable: false,       // No selection
  hasControls: false,      // No scaling/rotation controls
  hasBorders: false,       // No bounding box borders
  hasRotatingPoint: false, // No rotating point
});
canvas.add(rect);

// Disable canvas-wide selection if not needed
canvas.selection = false;
```

### 2.2. Leverage Object Caching

Enable `objectCaching` for complex objects (e.g., paths, filtered images, groups) to pre-rasterize them, avoiding repeated expensive drawing operations.

❌ BAD: Complex object without caching
```typescript
import { Path } from 'fabric';
const complexPath = new Path('M 0 0 L 100 0 L 50 100 z', { fill: 'purple' });
canvas.add(complexPath); // Rerendered every time
```

✅ GOOD: Enable object caching for performance
```typescript
import { Path } from 'fabric';
const complexPath = new Path('M 0 0 L 100 0 L 50 100 z', {
  fill: 'purple',
  objectCaching: true, // Cache complex path as a bitmap
});
canvas.add(complexPath);
```

### 2.3. Optimize Batch Operations

When adding or removing many objects, disable `renderOnAddRemove` to prevent unnecessary re-renders for each operation. Manually call `canvas.renderAll()` once after all changes.

❌ BAD: Multiple renders for batch operations
```typescript
for (let i = 0; i < 100; i++) {
  canvas.add(new Rect({ left: i * 2, top: i * 2, width: 10, height: 10 }));
} // canvas.renderAll() called 100 times
```

✅ GOOD: Batch operations with single render
```typescript
canvas.renderOnAddRemove = false; // Disable auto-render
for (let i = 0; i < 100; i++) {
  canvas.add(new Rect({ left: i * 2, top: i * 2, width: 10, height: 10 }));
}
canvas.renderOnAddRemove = true; // Re-enable if needed later
canvas.renderAll(); // Render once after all additions
```

### 2.4. Manage Object Visibility

Set `object.visible = false` for objects outside the current viewport or not intended to be seen, preventing them from being drawn.

```typescript
const offScreenRect = new Rect({ left: 1000, top: 1000, width: 50, height: 50 });
offScreenRect.visible = false; // Prevents rendering when off-screen
canvas.add(offScreenRect);
```

## 3. Common Patterns and Anti-patterns

### 3.1. Use Modern Accessors

Always use `getScaledWidth()` and `getScaledHeight()` to retrieve an object's dimensions, as `getWidth()` and `getHeight()` are deprecated.

❌ BAD: Using deprecated accessors
```typescript
const rect = new Rect({ width: 100, height: 50, scaleX: 2 });
console.log(rect.getWidth()); // Throws error in v2+
```

✅ GOOD: Using modern accessors
```typescript
const rect = new Rect({ width: 100, height: 50, scaleX: 2 });
console.log(rect.getScaledWidth());  // Outputs 200
console.log(rect.getScaledHeight()); // Outputs 50
```

### 3.2. Prefer `fabric.Group` over `fabric.PathGroup`

`fabric.PathGroup` is obsolete. Use `fabric.Group` for all object grouping, including SVG imports.

❌ BAD: Using `fabric.PathGroup`
```typescript
// This class no longer exists in modern Fabric.js
const pathGroup = new fabric.PathGroup([path1, path2]);
```

✅ GOOD: Using `fabric.Group`
```typescript
import { Group, Rect, Circle } from 'fabric';
const rect = new Rect({ width: 50, height: 50, fill: 'red' });
const circle = new Circle({ radius: 25, fill: 'blue' });
const group = new Group([rect, circle], { left: 100, top: 100 });
canvas.add(group);
```

### 3.3. Correct Image Filter Application

`applyFilters` is now synchronous. Call `canvas.requestRenderAll()` (or `renderAll()`) after applying filters to update the canvas.

❌ BAD: Old async filter syntax
```typescript
// This will throw an error
image.applyFilters(canvas.renderAll.bind(canvas));
```

✅ GOOD: Modern sync filter application
```typescript
import { FabricImage, Image } from 'fabric';

FabricImage.fromURL('path/to/image.png', (img) => {
  img.filters.push(new Image.filters.Grayscale());
  img.applyFilters(); // Synchronous
  canvas.add(img);
  canvas.requestRenderAll(); // Request a render after filters are applied
});
```

### 3.4. Serialize Canvas State Reliably

Always use `canvas.toJSON()` for serialization. If you have custom properties, ensure they are included in `toObject` methods of your custom classes. Version your JSON schema for future compatibility.

```typescript
// Save canvas state
const jsonState = canvas.toJSON(['customProperty1', 'customProperty2']);
localStorage.setItem('canvasState', JSON.stringify(jsonState));

// Load canvas state
const savedState = JSON.parse(localStorage.getItem('canvasState'));
canvas.loadFromJSON(savedState, () => {
  canvas.renderAll();
  console.log('Canvas loaded from JSON');
});
```

## 4. Testing Approaches

### 4.1. Benchmark Rendering Performance

Regularly benchmark `canvas.renderAll()` to identify performance regressions, especially after significant changes or adding complex objects.

```typescript
console.time('renderDuration');
canvas.renderAll();
console.timeEnd('renderDuration'); // Log render time
```

### 4.2. Test on Both Canvas Types

If your application uses both `Canvas` and `StaticCanvas` (e.g., for editing vs. display modes), ensure all features and objects render correctly on both.