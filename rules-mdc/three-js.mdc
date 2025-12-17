---
description: Enforce modern, performant, and maintainable three-js development practices, focusing on WebGL error handling, efficient asset management, and render loop optimization.
globs: **/*.{js,jsx}
---
# three-js Best Practices

This guide establishes the definitive coding standards for three-js development, ensuring robust, performant, and maintainable 3D applications. Adhere to these rules to prevent common pitfalls and leverage three-js effectively.

## 1. WebGL Error Handling & Extension Support

Always treat WebGL as a low-level layer requiring explicit error checks. Never assume universal extension support.

### ✅ GOOD: Proactive Error Checking & Graceful Fallbacks

During development, actively check `gl.getError()` to catch issues beyond `OUT_OF_MEMORY` or `CONTEXT_LOST`. In production, avoid blocking calls like `getError()` and `getParameter()`. Query extension support and provide fallbacks.

```javascript
// Development-only error checking (e.g., in a debug utility)
function checkGLErrors(gl) {
  let error = gl.getError();
  while (error !== gl.NO_ERROR) {
    console.error(`WebGL Error: ${error}`);
    error = gl.getError();
  }
}

// Extension support check
const renderer = new THREE.WebGLRenderer();
const gl = renderer.getContext();

if (gl.getExtension('ANGLE_instanced_arrays')) {
  // Use instanced rendering
} else {
  console.warn('ANGLE_instanced_arrays not supported, falling back to non-instanced rendering.');
  // Implement fallback or disable feature
}
```

### ❌ BAD: Ignoring WebGL Errors & Blindly Using Extensions

```javascript
// No error checking, leading to silent failures
const geometry = new THREE.BufferGeometry();
// ... potentially invalid buffer setup ...
renderer.render(scene, camera); // Errors might occur but go unnoticed

// Using advanced features without checking support
const instancedMesh = new THREE.InstancedMesh(geometry, material, count); // Fails on unsupported hardware
```

## 2. Asset Management & Lazy Loading

Optimize initial load times and prevent frame stalls by loading large assets lazily and using the correct loaders.

### ✅ GOOD: Lazy Loading with Dedicated Loaders & Modular Structure

Load assets only when needed. Use `GLTFLoader`, `TextureLoader`, etc., and organize assets in a `public/` directory.

```javascript
// main.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export function loadModel(url, scene) {
  const loader = new GLTFLoader();
  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf) => {
        scene.add(gltf.scene);
        resolve(gltf.scene);
      },
      (xhr) => {
        console.log(`${(xhr.loaded / xhr.total) * 100}% loaded`);
      },
      (error) => {
        console.error('An error happened loading the model:', error);
        reject(error);
      }
    );
  });
}

// In your scene setup, call loadModel when the asset is about to be visible
// await loadModel('/public/models/my-asset.glb', scene);
```

### ❌ BAD: Synchronous Loading & Unorganized Assets

```javascript
// main.js - synchronous loading blocks the main thread
// This is a simplified example, actual synchronous loading is less common with modern loaders
// but conceptually, loading all assets at once without a loading screen is bad.
const texture = new THREE.TextureLoader().load('large_texture.jpg'); // Blocks until loaded
const model = new GLTFLoader().load('heavy_model.glb'); // Blocks until loaded
scene.add(model.scene);
// ... UI freezes ...
```

## 3. Performance Optimization

Minimize draw calls, reuse resources, and keep shaders efficient.

### ✅ GOOD: Instancing, Geometry Batching, Material Reuse, & Texture Atlases

For repetitive objects, use `InstancedMesh`. Combine static geometries where possible. Reuse materials across objects. Consolidate textures into atlases.

```javascript
// InstancedMesh for many identical objects
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshNormalMaterial();
const count = 1000;
const mesh = new THREE.InstancedMesh(geometry, material, count);

const matrix = new THREE.Matrix4();
for (let i = 0; i < count; i++) {
  matrix.setPosition(Math.random() * 100 - 50, Math.random() * 100 - 50, Math.random() * 100 - 50);
  mesh.setMatrixAt(i, matrix);
}
mesh.instanceMatrix.needsUpdate = true;
scene.add(mesh);

// Material Reuse
const sharedMaterial = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
const obj1 = new THREE.Mesh(new THREE.BoxGeometry(), sharedMaterial);
const obj2 = new THREE.Mesh(new THREE.SphereGeometry(), sharedMaterial);
```

### ❌ BAD: Excessive Draw Calls & Unique Materials

```javascript
// Creating individual meshes for many identical objects
for (let i = 0; i < 1000; i++) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshNormalMaterial(); // New material for each
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(Math.random() * 100 - 50, Math.random() * 100 - 50, Math.random() * 100 - 50);
  scene.add(mesh); // 1000 draw calls, 1000 materials
}
```

## 4. Resource Disposal

Prevent memory leaks by explicitly disposing of three-js objects when they are no longer needed.

### ✅ GOOD: Explicit Disposal

Always call `dispose()` on geometries, materials, textures, and render targets when removing them from the scene.

```javascript
function disposeObject(obj) {
  if (obj.geometry) {
    obj.geometry.dispose();
  }
  if (obj.material) {
    if (Array.isArray(obj.material)) {
      obj.material.forEach(material => material.dispose());
    } else {
      obj.material.dispose();
    }
  }
  if (obj.texture) { // For textures directly attached to objects
    obj.texture.dispose();
  }
  if (obj.parent) {
    obj.parent.remove(obj);
  }
}

// Example usage:
const meshToRemove = scene.getObjectByName('myDynamicMesh');
if (meshToRemove) {
  disposeObject(meshToRemove);
}
```

### ❌ BAD: Relying on Garbage Collection

```javascript
const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
scene.add(mesh);
// Later, just remove from scene without disposing resources
scene.remove(mesh); // Geometry and material still consume GPU memory
mesh = null; // Only JS reference is cleared, not GPU resources
```

## 5. Project Structure & Development Workflow

Adopt a modern development setup for efficient development and deployment.

### ✅ GOOD: npm + Vite & Modular Imports

Use npm for dependency management and Vite for local development and production builds. Structure your project with a clear entry point (`main.js`) and a `public/` folder for static assets.

```javascript
// index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>My three.js App</title>
</head>
<body>
    <script type="module" src="/main.js"></script>
</body>
</html>

// main.js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// ... rest of your scene setup
```

### ❌ BAD: CDN-only & Global Scope Pollution

```html
<!-- index.html -->
<script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/controls/OrbitControls.js"></script>
<script>
    // All code in global scope, no module benefits
    const scene = new THREE.Scene();
    // ...
</script>
```