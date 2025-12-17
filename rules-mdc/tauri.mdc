---
description: This guide provides definitive, actionable best practices for writing robust, performant, and secure Rust backend code in Tauri applications, focusing on modern patterns and common pitfalls.
globs: **/*.rs
---
# Tauri Best Practices (Rust Backend)

Tauri applications thrive on a lean, secure Rust backend. This guide outlines the essential patterns and anti-patterns for writing high-quality Rust code that integrates seamlessly with your frontend.

## 1. Code Organization and Structure

Organize your Rust code by feature, not by file type. Keep related structs, enums, and their `impl` blocks together within a module. Leverage Rust's module system for clear separation of concerns and encapsulation.

**✅ GOOD: Feature-based Modules**

```rust
// src-tauri/src/main.rs
mod user; // Defines the 'user' module

fn main() {
    // ...
}

// src-tauri/src/user/mod.rs
pub mod manager; // Exposes user::manager

// src-tauri/src/user/manager.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct User {
    pub id: u32,
    pub name: String,
}

impl User {
    pub fn new(id: u32, name: String) -> Self {
        Self { id, name }
    }

    pub fn greet(&self) -> String {
        format!("Hello, {}!", self.name)
    }
}
```

**❌ BAD: Separating Types and Implementations**

```rust
// src-tauri/src/types.rs (AVOID)
pub struct User { /* ... */ }

// src-tauri/src/user_impls.rs (AVOID)
impl User { /* ... */ }
```

> **Action**: Always run `cargo fmt` and `cargo clippy` to enforce standard Rust style and catch common errors.

## 2. Tauri Command Fundamentals

Tauri commands are the bridge between your frontend and Rust backend. Keep them focused, asynchronous, and delegate complex logic.

**✅ GOOD: Lean, Async Commands**

```rust
// src-tauri/src/main.rs
#[tauri::command]
async fn greet(name: String) -> Result<String, String> {
    // Delegate complex logic to a dedicated service/manager
    let greeting = user::manager::User::new(0, name).greet();
    Ok(greeting)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**❌ BAD: Blocking or Overly Complex Commands**

```rust
// src-tauri/src/main.rs (AVOID)
#[tauri::command]
fn process_large_file_sync(path: String) -> Result<String, String> {
    // This blocks the main thread, making the UI unresponsive
    std::thread::sleep(std::time::Duration::from_secs(5));
    Ok(format!("Processed {}", path))
}
```

## 3. Manage Global State with `tauri::State`

For shared, mutable application state, `tauri::State` is the definitive solution. Wrap your state in `Arc<Mutex<T>>` or `Arc<RwLock<T>>` for thread-safe access.

**✅ GOOD: Thread-Safe Global State**

```rust
// src-tauri/src/main.rs
use std::sync::{Arc, Mutex};

struct AppState {
    counter: u32,
}

#[tauri::command]
fn increment_counter(state: tauri::State<'_, Arc<Mutex<AppState>>>) -> Result<u32, String> {
    let mut app_state = state.lock().map_err(|e| format!("State lock error: {}", e))?;
    app_state.counter += 1;
    Ok(app_state.counter)
}

fn main() {
    let app_state = Arc::new(Mutex::new(AppState { counter: 0 }));
    tauri::Builder::default()
        .manage(app_state) // Register the state
        .invoke_handler(tauri::generate_handler![increment_counter])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**❌ BAD: Unmanaged Global Statics**

```rust
// src-tauri/src/main.rs (AVOID)
static mut GLOBAL_COUNTER: u32 = 0; // NOT thread-safe, prone to data races
#[tauri::command]
fn get_global_counter() -> u32 {
    unsafe { GLOBAL_COUNTER } // Unsafe code, difficult to reason about
}
```

## 4. Robust Error Handling

Use `thiserror` or `anyhow` for structured error handling. Always return `Result<T, E>` from commands, where `E` is a serializable error type. This allows the frontend to display meaningful error messages.

**✅ GOOD: Custom, Serializable Errors**

```rust
// src-tauri/src/error.rs
use thiserror::Error;
use serde::Serialize;

#[derive(Debug, Error, Serialize)]
pub enum AppError {
    #[error("Resource not found: {0}")]
    NotFound(String),
    #[error("Database operation failed: {0}")]
    Database(String),
    #[error("Internal server error: {0}")]
    Internal(String),
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Internal(format!("IO error: {}", err))
    }
}

// src-tauri/src/main.rs
use tauri::async_runtime::spawn_blocking;
use crate::error::AppError; // Assuming error.rs is in src

#[tauri::command]
async fn read_config_file(path: String) -> Result<String, AppError> {
    // Use spawn_blocking for potentially blocking I/O
    spawn_blocking(move || {
        std::fs::read_to_string(&path).map_err(|e| AppError::from(e))
    }).await.map_err(|e| AppError::Internal(format!("Blocking task failed: {}", e)))?
}
```

**❌ BAD: Generic Error Strings or Panics**

```rust
// src-tauri/src/main.rs (AVOID)
#[tauri::command]
fn divide(a: u32, b: u32) -> Result<u32, String> {
    if b == 0 {
        return Err("Cannot divide by zero".to_string()); // Generic string
    }
    Ok(a / b)
}
#[tauri::command]
fn dangerous_operation() {
    panic!("Something went wrong!"); // Crashes the backend
}
```

## 5. Performance Considerations

Tauri's multi-process model means the Rust core can run heavy tasks without freezing the UI. Leverage `async/await` and `spawn_blocking` for optimal responsiveness.

**✅ GOOD: Offload Blocking Tasks**

```rust
// src-tauri/src/main.rs
use tauri::async_runtime::spawn_blocking;

#[tauri::command]
async fn perform_heavy_computation() -> Result<u64, String> {
    let result = spawn_blocking(|| {
        // Simulate a CPU-bound blocking task
        let mut sum = 0;
        for i in 0..1_000_000; {
            sum += i;
        }
        sum
    }).await.map_err(|e| e.to_string())?;
    Ok(result)
}
```

**❌ BAD: Blocking the Async Runtime**

```rust
// src-tauri/src/main.rs (AVOID)
#[tauri::command]
async fn sync_network_call() -> Result<String, String> {
    // This blocks the async runtime, impacting other async tasks
    let response = reqwest::blocking::get("http://example.com")
        .map_err(|e| e.to_string())?
        .text()
        .map_err(|e| e.to_string())?;
    Ok(response)
}
```

## 6. Common Pitfalls and Gotchas

### Security: Principle of Least Privilege

**Opinion**: Explicitly define capabilities in `src-tauri/tauri.conf.json`. Never grant wildcard access (`*`) unless absolutely necessary for specific, audited plugins.

**✅ GOOD: Granular Capabilities (`src-tauri/tauri.conf.json`)**

```json
{
  "tauri": {
    "allowlist": {
      "fs": {
        "readFile": true,
        "writeFile": true,
        "scope": ["$APPCONFIG/*", "$APPDATA/*"]
      },
      "shell": {
        "open": true
      }
    },
    "security": {
      "csp": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline';"
    }
  }
}
```

**❌ BAD: Overly Permissive Capabilities (`src-tauri/tauri.conf.json`)**

```json
{
  "tauri": {
    "allowlist": {
      "all": true // Grants access to ALL APIs, highly insecure
    }
  }
}
```

### Frontend vs. Backend Logic

**Opinion**: Keep sensitive data and business logic exclusively in the Rust backend. The frontend (WebView) should only handle UI presentation and user interaction.

**✅ GOOD: Rust for Sensitive Logic**

```rust
// src-tauri/src/main.rs
#[tauri::command]
async fn process_payment(amount: f64, token: String) -> Result<String, AppError> {
    // Payment processing logic, interacts with secure APIs
    // ...
    Ok("Payment successful".into())
}
```

**❌ BAD: Handling Secrets in Frontend JavaScript**

```javascript
// src/main.js (AVOID)
async function sendPayment() {
  const secretApiKey = "YOUR_HARDCODED_API_KEY"; // EXPOSED!
  // ... send payment directly from frontend
}
```

## 7. Testing Approaches

Unit test your Rust modules independently. For commands, consider integration tests that simulate frontend calls.

**✅ GOOD: Unit Testing Rust Logic**

```rust
// src-tauri/src/user/manager.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_greeting() {
        let user = User::new(1, "Alice".into());
        assert_eq!(user.greet(), "Hello, Alice!");
    }

    #[test]
    fn test_user_id() {
        let user = User::new(42, "Bob".into());
        assert_eq!(user.id, 42);
    }
}
```