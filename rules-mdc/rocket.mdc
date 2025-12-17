---
description: Definitive guidelines for building robust, performant, and maintainable backend services with Rocket 0.5+, focusing on modern Rust async patterns, API design, and testing.
globs: **/*.rs
---
# rocket Best Practices

This document outlines the definitive best practices for developing with Rocket, Rust's async web framework. Adhere to these guidelines to ensure your Rocket applications are performant, secure, and easily maintainable.

## 1. Code Organization and Structure

Organize your codebase by feature, not by file type. Each logical feature (e.g., `auth`, `posts`, `users`) should reside in its own module, encapsulating its routes, data structures, and business logic.

### ❌ BAD: Anemic `types.rs` and `impl.rs`

```rust
// src/types.rs
pub struct User { /* ... */ }
pub struct Post { /* ... */ }

// src/impls.rs
impl User { /* ... */ }
impl Post { /* ... */ }

// src/routes.rs
#[get("/users")]
fn get_users() { /* ... */ }
```

### ✅ GOOD: Feature-based Modules

```rust
// src/main.rs
mod auth;
mod posts;
mod users;

#[rocket::main]
async fn main() -> Result<(), rocket::Error> {
    rocket::build()
        .mount("/auth", auth::routes())
        .mount("/posts", posts::routes())
        .mount("/users", users::routes())
        .launch()
        .await
}

// src/users/mod.rs
pub mod models;
pub mod handlers;

pub fn routes() -> Vec<rocket::Route> {
    rocket::routes![
        handlers::get_all_users,
        handlers::get_user_by_id,
        // ...
    ]
}

// src/users/models.rs
pub struct User {
    pub id: i32,
    pub username: String,
}

// src/users/handlers.rs
use super::models::User;
use rocket::serde::json::Json;

#[get("/")]
pub async fn get_all_users() -> Json<Vec<User>> {
    // Fetch users from DB
    Json(vec![User { id: 1, username: "test".to_string() }])
}

#[get("/<id>")]
pub async fn get_user_by_id(id: i32) -> Option<Json<User>> {
    // Fetch user from DB
    if id == 1 {
        Some(Json(User { id: 1, username: "test".to_string() }))
    } else {
        None
    }
}
```

## 2. API Design

Design APIs with clarity, explicitness, and robust error handling.

### 2.1 Declarative Routing

Leverage Rocket's attribute macros for clear, concise route definitions.

```rust
// Define a GET route at /hello/<name>
#[get("/hello/<name>")]
async fn hello(name: String) -> String {
    format!("Hello, {}!", name)
}

// Define a POST route at /users with JSON body
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
struct NewUser {
    username: String,
    email: String,
}

#[post("/users", data = "<user>")]
async fn create_user(user: Json<NewUser>) -> Json<NewUser> {
    // Save user to DB
    user
}
```

### 2.2 Request Guards for Validation and Authentication

Use custom request guards to centralize input validation, authentication, and authorization. This keeps handler logic clean and ensures consistent policy enforcement.

```rust
use rocket::request::{self, Request, FromRequest};
use rocket::outcome::Outcome;
use rocket::http::Status;

pub struct AdminUser;

#[rocket::async_trait]
impl<'r> FromRequest<'r> for AdminUser {
    type Error = ();

    async fn from_request(request: &'r Request<'_>) -> request::Outcome<Self, Self::Error> {
        // Example: Check for a specific header or session token
        if request.headers().contains("X-Admin-Token") {
            Outcome::Success(AdminUser)
        } else {
            Outcome::Failure((Status::Unauthorized, ()))
        }
    }
}

#[get("/admin")]
async fn admin_panel(admin: AdminUser) -> &'static str {
    "Welcome, administrator!"
}
```

### 2.3 Responders for Consistent Responses

Implement `Responder` for custom response types to standardize API output, especially for errors.

```rust
use rocket::http::{Status, ContentType};
use rocket::response::{self, Responder, Response};
use std::io::Cursor;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub code: u16,
    pub message: String,
}

impl<'r, 'o: 'r> Responder<'r, 'o> for ApiError {
    fn respond_to(self, _: &'r Request<'_>) -> response::Result<'o> {
        let json = serde_json::to_string(&self).unwrap();
        Response::build()
            .sized_body(self.message.len(), Cursor::new(json))
            .header(ContentType::new("application", "json"))
            .status(Status::new(self.code))
            .ok()
    }
}

#[get("/fail")]
async fn fail_route() -> Result<&'static str, ApiError> {
    Err(ApiError {
        code: 400,
        message: "Something went wrong!".to_string(),
    })
}
```

## 3. Common Patterns and Anti-patterns

### 3.1 Async Everywhere

Rocket 0.5+ is fully async. All handlers and I/O operations (database, templates) *must* be `async fn`.

### ❌ BAD: Blocking I/O in handlers

```rust
// This will block the Rocket worker thread
#[get("/")]
fn index() -> String {
    std::thread::sleep(std::time::Duration::from_secs(1));
    "Hello, world!".to_string()
}
```

### ✅ GOOD: Async I/O with `await`

```rust
use rocket_db_pools::{sqlx, Database};

#[Database("my_db")]
struct MyDatabase(sqlx::PgPool);

#[get("/")]
async fn index(mut db: Connection<MyDatabase>) -> String {
    // Perform async DB operation
    sqlx::query!("SELECT 1").fetch_one(&mut *db).await.unwrap();
    "Hello, async world!".to_string()
}
```

### 3.2 `#[rocket::async_trait]` for Trait Implementations

When implementing Rocket traits (e.g., `FromRequest`), always use `#[rocket::async_trait]`.

### ❌ BAD: Missing `async_trait`

```rust
// Will result in a compiler error
impl<'r> FromRequest<'r> for MyType {
    type Error = ();
    async fn from_request(...) { /* ... */ }
}
```

### ✅ GOOD: Correct `async_trait` usage

```rust
#[rocket::async_trait]
impl<'r> FromRequest<'r> for MyType {
    type Error = ();
    async fn from_request(...) { /* ... */ }
}
```

### 3.3 Form and JSON Validation

Use `FromForm` for URL-encoded data and `Json` with `serde::Deserialize` for JSON. For strict form validation, wrap fields in `Strict<T>`.

```rust
use rocket::form::{Form, Strict};

#[derive(FromForm)]
struct UserProfile {
    name: String,
    #[field(validate = len(min = 10))]
    bio: String,
    age: Strict<u8>, // Age must be present and valid
}

#[post("/profile", data = "<profile>")]
async fn update_profile(profile: Form<UserProfile>) -> String {
    format!("Profile updated for {}", profile.name)
}
```

## 4. Error Handling

Always return `Result<T, E>` from handlers where `E` is a custom error type that implements `Responder`. This provides clear error messages and HTTP status codes.

```rust
// src/errors.rs
use super::api_response::ApiError; // Assuming ApiError from section 2.3

pub enum MyHandlerError {
    NotFound,
    InvalidInput(String),
    DatabaseError(String),
}

impl From<MyHandlerError> for ApiError {
    fn from(err: MyHandlerError) -> Self {
        match err {
            MyHandlerError::NotFound => ApiError { code: 404, message: "Resource not found".to_string() },
            MyHandlerError::InvalidInput(msg) => ApiError { code: 400, message: format!("Invalid input: {}", msg) },
            MyHandlerError::DatabaseError(msg) => ApiError { code: 500, message: format!("Database error: {}", msg) },
        }
    }
}

// src/handlers.rs
use crate::errors::MyHandlerError;
use crate::api_response::ApiError;

#[get("/item/<id>")]
async fn get_item(id: i32) -> Result<String, ApiError> {
    if id == 0 {
        return Err(MyHandlerError::NotFound.into());
    }
    if id < 0 {
        return Err(MyHandlerError::InvalidInput("ID cannot be negative".to_string()).into());
    }
    Ok(format!("Item {}", id))
}
```

## 5. Performance Considerations

### 5.1 Database Connection Pooling

Use `rocket_db_pools` for efficient, non-blocking database access. Configure pools in `Rocket.toml`.

```rust
// Cargo.toml
// rocket_db_pools = { version = "0.1.0", features = ["sqlx", "postgres"] }

// Rocket.toml
[default.databases.my_db]
url = "postgres://user:pass@host/db"

// src/main.rs
use rocket_db_pools::{sqlx, Database, Connection};

#[Database("my_db")]
struct MyDatabase(sqlx::PgPool);

#[get("/count")]
async fn get_count(mut db: Connection<MyDatabase>) -> String {
    let count: i64 = sqlx::query_scalar!("SELECT COUNT(*) FROM items")
        .fetch_one(&mut *db)
        .await
        .unwrap_or_default();
    format!("Total items: {}", count)
}
```

## 6. Security Best Practices

### 6.1 Configuration via `Rocket.toml`

Never hardcode sensitive information (database URLs, API keys) in your code. Use `Rocket.toml` for environment-specific settings. Version control only a template of `Rocket.toml` and use environment variables for actual secrets.

```toml
# Rocket.toml
[default]
address = "0.0.0.0"
port = 8000
secret_key = { env = "ROCKET_SECRET_KEY" } # Load from env var

[default.databases.my_db]
url = { env = "DATABASE_URL" }
```

## 7. Testing Approaches

### 7.1 Integration Tests with `local::asynchronous::Client`

Write comprehensive integration tests that spin up an in-process Rocket instance. This ensures your routes, guards, and responders work together as expected.

```rust
// tests/integration_test.rs
use rocket::local::asynchronous::Client;
use rocket::{build, routes};

#[rocket::async_test]
async fn test_hello_world() {
    let rocket = build().mount("/", routes![crate::hello]); // Assuming `hello` from section 2.1
    let client = Client::tracked(rocket).await.expect("valid rocket instance");
    let response = client.get("/hello/world").dispatch().await;

    assert_eq!(response.status(), rocket::http::Status::Ok);
    assert_eq!(response.into_string().await, Some("Hello, world!".to_string()));
}
```

### 7.2 CI Pipeline Checks

Integrate `cargo fmt -- --check`, `cargo clippy -- -D warnings`, and `cargo test --all-features` into your CI pipeline to enforce code quality and catch common issues automatically.