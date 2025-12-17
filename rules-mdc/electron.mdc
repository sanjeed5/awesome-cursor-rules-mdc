---
description: This guide provides opinionated, actionable best practices for building secure, performant, and maintainable Electron applications using modern patterns and consistent tooling.
globs: **/*.{js,jsx}
---
# Electron Best Practices

Electron development demands a disciplined approach to security, performance, and maintainability. This guide outlines the definitive best practices for our team, leveraging modern Electron features and tooling (targeting Electron 28+).

## 1. Project Setup & Structure

Always start with **Electron Forge** to standardize project structure, build pipelines, and stay aligned with the latest Electron APIs.

-   **Scaffolding:** Use a modern template like Vite + TypeScript.
    ❌ BAD: Manual setup, outdated CLIs.
    ✅ GOOD:
    ```bash
    npx create-electron-app@latest my-app --template=vite-typescript
    ```

-   **File Naming:** Adhere to Electron's coding style for JavaScript files.
    ❌ BAD: `my_module.js`
    ✅ GOOD: `my-module.js`

## 2. Security Fundamentals (Non-Negotiable)

Security is paramount. Always enable context isolation and expose APIs safely.

-   **Context Isolation (Mandatory):** Keep `contextIsolation` enabled. It's on by default since Electron 12.
    ❌ BAD: `new BrowserWindow({ webPreferences: { contextIsolation: false } })`
    ✅ GOOD: (Default behavior, no explicit setting needed unless overriding)
    ```javascript
    // main.mjs
    new BrowserWindow({
      webPreferences: {
        preload: path.join(__dirname, 'preload.mjs'),
        sandbox: true // Strongly recommended
      }
    })
    ```

-   **Safe API Exposure with `contextBridge`:** Never mutate the global `window` object directly. Use `contextBridge.exposeInMainWorld` and filter arguments.
    ❌ BAD: Exposing `ipcRenderer.send` directly.
    ```javascript
    // preload.mjs
    const { contextBridge, ipcRenderer } = require('electron');
    contextBridge.exposeInMainWorld('myAPI', {
      send: ipcRenderer.send // ❌ Allows renderer to send arbitrary IPC messages
    });
    ```
    ✅ GOOD: Expose specific, argument-filtered functions.
    ```javascript
    // preload.mjs
    import { contextBridge, ipcRenderer } from 'electron';

    contextBridge.exposeInMainWorld('electronAPI', {
      loadPreferences: () => ipcRenderer.invoke('load-prefs'),
      saveSettings: (settings) => {
        // ✅ Validate and filter arguments before sending
        if (typeof settings === 'object' && settings !== null) {
          ipcRenderer.send('save-settings', settings);
        } else {
          console.error('Invalid settings object provided.');
        }
      }
    });

    // interface.d.ts (for TypeScript)
    declare global {
      interface Window {
        electronAPI: {
          loadPreferences: () => Promise<any>;
          saveSettings: (settings: object) => void;
        };
      }
    }
    ```

-   **Content Security Policy (CSP):** Implement a strict CSP in your `index.html` or via `webRequest.onHeadersReceived`.
    ✅ GOOD:
    ```html
    <!-- index.html -->
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'">
    ```

## 3. ES Modules (ESM) Adoption (Electron 28+)

Leverage native ESM for cleaner, more modern code.

-   **Main Process:** Use `.mjs` extension or `"type": "module"` in `package.json`.
    -   **`await` for Pre-Ready APIs:** ESM imports are asynchronous. Ensure critical APIs (e.g., `app.setPath`) are `await`ed before `app.whenReady()`.
    ❌ BAD:
    ```javascript
    // main.mjs
    import './setup-paths.mjs'; // May resolve after app is ready
    app.whenReady().then(() => { /* ... */ });
    ```
    ✅ GOOD:
    ```javascript
    // main.mjs
    import { app } from 'electron';
    await import('./setup-paths.mjs'); // Guarantees execution before ready
    app.whenReady().then(() => { /* ... */ });
    ```

-   **Preload Scripts:** Always use the `.mjs` extension for ESM preload scripts.
    -   **Sandboxed Preloads:** Cannot use ESM imports. Bundle them if external modules are needed.
    -   **Context Isolation:** Required for dynamic Node.js ESM imports in unsandboxed preloads.
    ❌ BAD: `preload.js` with `import` statements.
    ✅ GOOD: `preload.mjs`

## 4. IPC Communication

Use `ipcMain.handle` and `ipcRenderer.invoke` for explicit request-response patterns.

-   **Request-Response:**
    ❌ BAD: Using `ipcRenderer.send` for requests that expect a response.
    ```javascript
    // renderer.js
    ipcRenderer.send('get-data', someId);
    ipcRenderer.on('data-response', (event, data) => { /* ... */ }); // Race condition prone
    ```
    ✅ GOOD:
    ```javascript
    // main.mjs
    ipcMain.handle('get-data', async (event, someId) => {
      // ✅ Perform validation, access native APIs
      return await fetchData(someId);
    });

    // renderer.js
    const data = await window.electronAPI.getData(someId); // Assuming exposed via contextBridge
    ```

## 5. System Path Handling

Always use Node.js `path` and `os` modules for cross-platform compatibility.

-   **File Paths:** Use `path.join()` for concatenation.
    ❌ BAD: `app.getPath('userData') + '/config.json'`
    ✅ GOOD:
    ```javascript
    import path from 'node:path';
    import { app } from 'electron';
    const configPath = path.join(app.getPath('userData'), 'config.json');
    ```

-   **Temporary Directories:** Use `os.tmpdir()`.
    ❌ BAD: `'/tmp/my-app-data'`
    ✅ GOOD:
    ```javascript
    import os from 'node:os';
    const tempDir = os.tmpdir();
    ```

## 6. Testing & Linting

Integrate Electron's built-in tooling for consistent code quality.

-   **Linting:** Run `npm run lint` regularly and integrate into pre-commit hooks.
    ✅ GOOD: Ensure your `package.json` includes:
    ```json
    "scripts": {
      "lint": "electron-builder lint" // Or specific linter like 'eslint .'
    }
    ```

-   **Unit Tests:** Add new tests for any changes or new features.
    ✅ GOOD: `npm run test`
    ```json
    "scripts": {
      "test": "electron-mocha spec" // Example with electron-mocha
    }
    ```

## 7. Staying Current

Electron evolves rapidly. Proactively manage updates and breaking changes.

-   **Official Documentation:** Always consult the version-specific official documentation.
-   **Breaking Changes:** Regularly review the "Breaking Changes" page for each major Electron release to anticipate necessary updates.