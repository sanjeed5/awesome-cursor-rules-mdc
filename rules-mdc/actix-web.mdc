---
description: Definitive guidelines for building high-performance, maintainable, and secure web applications with actix-web 4, focusing on modern Rust best practices.
globs: **/*.rs
---
# actix-web Best Practices

Actix-web 4 is the premier choice for high-performance Rust web services. Adhere to these guidelines for clean, scalable, and production-ready code. Always run `cargo fmt` and `cargo clippy` before committing.

## 1. Code Organization & Structure

Organize your codebase by feature or domain, not by generic file types. This improves discoverability and maintainability.

### ✅ GOOD: Modular by Feature

```rust
// src/main.rs
mod db; // Database access logic
mod routes; // HTTP route handlers, grouped by resource
mod services; // Business logic layer
mod models; // Data structures (e.g., database models, DTOs)

use actix_web::{web, App, HttpServer};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging early
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    let db_pool = db::init_pool().await
        .expect("Failed to create DB pool");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(db_pool.clone())) // Inject shared DB pool
            .service(
                web::scope("/api/v1") // API versioning and grouping
                    .configure(routes::users::config) // User-related routes
                    .configure(routes::products::config) // Product-related routes
            )
            // Add global middleware here (e.g., Logger, Compress)
            .wrap(actix_web::middleware::Logger::default())
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

// src/routes/users.rs
use actix_web::{web, HttpResponse, Responder};
use crate::db::PgPool; // Assuming PgPool is public in db module
use crate::services::users as user_service; // Business logic layer

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/users").route(web::get().to(get_all_users)));
    cfg.service(web::resource("/users/{id}").route(web::get().to(get_user_by_id)));
}

async fn get_all_users(pool: web::Data<PgPool>) -> impl Responder {
    match user_service::find_all_users(&pool).await {
        Ok(users) => HttpResponse::Ok().json(users),
        Err(e) => {
            log::error!("Failed to fetch users: {:?}", e);
            HttpResponse::InternalServerError().finish()
        },
    }
}

// src/services/users.rs
use crate::db::PgPool;
use crate::models::User; // Assuming User model

pub async fn find_all_users(pool: &PgPool) -> Result<Vec<User>, sqlx::Error> {
    sqlx::query_as::<_, User>("SELECT id, name, email FROM users")
        .fetch_all(pool)
        .await
}
```

### ❌ BAD: Monolithic `main.rs` or `types.rs`/`impl.rs` for everything

```rust
// src/main.rs (all routes, models, db logic in one file)
// src/types.rs (all structs, regardless of domain)
// src/impls.rs (all impl blocks, regardless of domain)
```
This approach quickly becomes unmanageable and hinders collaboration.

## 2. Application State (`web::Data<T>`)

Use `web::Data<T>` for dependency injection and sharing application-wide state. For mutable state, wrap it in `Arc<Mutex<T>>` or `Arc<RwLock<T>>` and ensure it's initialized *outside* the `HttpServer::new` closure.

### ✅ GOOD: Shared Immutable and Mutable State

```rust
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use std::sync::{Arc, Mutex};

struct AppState {
    app_name: String,
    request_count: Arc<Mutex<usize>>, // Shared mutable state
}

async fn index(data: web::Data<AppState>) -> impl Responder {
    let mut count = data.request_count.lock().unwrap();
    *count += 1;
    HttpResponse::Ok().body(format!(
        "Hello from {}! Request count: {}",
        data.app_name, count
    ))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize shared mutable state *outside* the closure
    let request_count = Arc::new(Mutex::new(0));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(AppState {
                app_name: "My Actix App".to_string(),
                request_count: request_count.clone(), // Clone Arc for each worker
            }))
            .route("/", web::get().to(index))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
```

### ❌ BAD: Mutable state initialized inside `HttpServer::new`

```rust
// This will create a *new* Mutex for each worker thread,
// leading to desynchronized counts.
HttpServer::new(move || {
    let request_count = Arc::new(Mutex::new(0)); // BAD: Each worker gets its own counter
    App::new()
        .app_data(web::Data::new(AppState {
            app_name: "My Actix App".to_string(),
            request_count: request_count.clone(),
        }))
        .route("/", web::get().to(index))
})
```

## 3. Extractors & Responders

Leverage Actix-web's powerful, type-safe extractors (`web::Json`, `web::Path`, `web::Query`) and ensure your handlers return types that implement `Responder`.

### ✅ GOOD: Type-Safe Request/Response Handling

```rust
use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct UserInfo {
    name: String,
    email: String,
}

#[derive(Serialize)]
struct UserResponse {
    id: u32,
    name: String,
}

// Handler with Path, Json extractors and Json response
async fn create_user(
    path: web::Path<u32>, // Extracts from /users/{id}
    user_info: web::Json<UserInfo>, // Extracts and deserializes JSON body
) -> impl Responder {
    let user_id = path.into_inner();
    log::info!("Creating user {} with name: {}", user_id, user_info.name);
    // ... business logic ...
    HttpResponse::Created().json(UserResponse {
        id: user_id,
        name: user_info.name.clone(),
    })
}
```

### ❌ BAD: Manual Request Body Parsing

```rust
// Avoid manually reading and deserializing request bodies; it's error-prone and verbose.
async fn create_user_bad(mut payload: web::Payload) -> Result<HttpResponse, actix_web::Error> {
    use futures::StreamExt;
    let mut body = web::BytesMut::new();
    while let Some(chunk) = payload.next().await {
        body.extend_from_slice(&chunk?)
    }
    let user_info: UserInfo = serde_json::from_slice(&body)
        .map_err(actix_web::error::ErrorBadRequest)?; // Manual deserialization
    // ...
    Ok(HttpResponse::Ok().finish())
}
```

## 4. Error Handling

Define custom error types that implement `ResponseError` for consistent, structured error responses. Use `thiserror` for ergonomic error definition.

### ✅ GOOD: Custom Error Types with `ResponseError`

```rust
use actix_web::{error::ResponseError, http::StatusCode, HttpResponse};
use derive_more::{Display, Error};
use serde::Serialize;
use crate::models::User; // Assuming User model
use crate::db::PgPool; // Assuming PgPool

#[derive(Debug, Display, Error, Serialize)]
#[display(fmt = "{}", message)]
pub enum AppError {
    #[display(fmt = "User not found: {}", id)]
    UserNotFound { id: u32, message: String },
    #[display(fmt = "Database error: {}", _0)]
    DbError(#[from] sqlx::Error),
    #[display(fmt = "Validation error: {}", message)]
    ValidationError { message: String },
    #[display(fmt = "Internal server error")]
    InternalError,
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        let status_code = match self {
            AppError::UserNotFound { .. } => StatusCode::NOT_FOUND,
            AppError::DbError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::ValidationError { .. } => StatusCode::BAD_REQUEST,
            AppError::InternalError => StatusCode::INTERNAL_SERVER_ERROR,
        };
        HttpResponse::build(status_code).json(self) // Serialize error to JSON
    }
}

// Example handler using custom error
async fn get_user_handler(
    path: web::Path<u32>,
    pool: web::Data<PgPool>,
) -> Result<HttpResponse, AppError> {
    let user_id = path.into_inner();
    let user = sqlx::query_as::<_, User>("SELECT id, name, email FROM users WHERE id = $1")
        .bind(user_id)
        .fetch_optional(pool.get_ref())
        .await? // Converts sqlx::Error to AppError::DbError
        .ok_or(AppError::UserNotFound { id: user_id, message: format!("User with ID {} not found", user_id) })?;

    Ok(HttpResponse::Ok().json(user))
}
```

### ❌ BAD: Generic `actix_web::Error` or `panic!`

```rust
// Returns a generic 500 error without specific context, making debugging hard.
async fn get_user_bad(path: web::Path<u32>) -> Result<HttpResponse, actix_web::Error> {
    let user_id = path.into_inner();
    if user_id == 0 {
        return Err(actix_web::error::ErrorBadRequest("Invalid ID"));
    }
    // ...
    Ok(HttpResponse::Ok().finish())
}

// Panicking in a handler is unacceptable for production; it crashes the worker.
async fn get_user_panic() -> impl Responder {
    panic!("Something went terribly wrong!");
}
```

## 5. Middleware & Security

Employ `actix-web`'s robust middleware for logging, compression, CORS, and other cross-cutting concerns. Always include `Logger` for request tracing.

### ✅ GOOD: Essential Middleware Stack

```rust
use actix_web::{middleware, App, HttpServer};
use actix_cors::Cors;
use env_logger::Env;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin() // Configure specific origins for production
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(middleware::Logger::default()) // Request logging
            .wrap(middleware::Compress::default()) // Response compression
            .wrap(middleware::NormalizePath::trim()) // Trailing slash handling
            .wrap(cors) // Cross-Origin Resource Sharing
            // ... your routes ...
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
```

### ❌ BAD: Missing Logging or Basic Security

```rust
// Without Logger, debugging production issues is significantly harder.
// Without CORS, frontend apps on different origins will fail.
HttpServer::new(move || {
    App::new()
        // No .wrap(middleware::Logger::default())
        // No .wrap(Cors::default())
        // ...
})
```

## 6. Performance & Blocking Operations

Actix-web is asynchronous. Avoid blocking I/O or CPU-bound tasks directly in handlers. Use `web::block` for blocking operations or offload to a dedicated thread pool.

### ✅ GOOD: Offloading Blocking Operations

```rust
use actix_web::{web, HttpResponse, Responder};
use std::thread;
use std::time::Duration;

// A simulated blocking DB call or heavy computation
fn blocking_task(input: String) -> String {
    thread::sleep(Duration::from_secs(1)); // Simulate blocking
    format!("Processed: {}", input)
}

async fn handle_blocking_request(body: web::Bytes) -> impl Responder {
    let input = String::from_utf8(body.to_vec()).unwrap_or_default();
    // Use web::block to run the blocking task on a separate thread pool
    let result = web::block(move || blocking_task(input)).await;

    match result {
        Ok(output) => HttpResponse::Ok().body(output),
        Err(e) => {