---
description: Definitive guidelines for writing idiomatic, performant, and maintainable Rust code, focusing on modern best practices for structure, patterns, and error handling.
globs: **/*
---
# Rust Best Practices

This guide outlines our team's definitive best practices for writing Rust code. Adhere to these principles to ensure consistency, performance, and maintainability across all projects.

## 1. Code Organization and Structure

### 1.1 Module Structure: Feature-Driven, Not File-Type-Driven
Organize modules by *feature or functionality*, not by separating types from implementations. Keep related structs, enums, and their `impl` blocks together within the same module.

❌ BAD:
```rust
// types.rs
pub struct User { pub id: u64 }
// impls.rs
impl User {
    pub fn new(id: u64) -> Self { Self { id } }
}
```

✅ GOOD:
```rust
// user.rs (or a user/mod.rs)
pub struct User { pub id: u64 }
impl User {
    pub fn new(id: u64) -> Self { Self { id } }
    pub fn get_id(&self) -> u64 { self.id }
}
```

### 1.2 Small, Cohesive Structs
Design structs to be small and focused. Break down complex data into multiple smaller, composable structs. This improves borrow-checker ergonomics and API flexibility.

❌ BAD:
```rust
pub struct OrderProcessor {
    pub order_id: u64, pub customer_id: u64, pub items: Vec<String>,
    pub total_amount: f64, pub payment_status: String,
    pub shipping_address_street: String, pub shipping_address_city: String,
}
```

✅ GOOD:
```rust
pub struct Order {
    pub id: u64, pub customer_id: u64, pub items: Vec<String>,
    pub total_amount: f64, pub payment_status: String,
}
pub struct Address { pub street: String, pub city: String }
pub struct OrderProcessor {
    pub order: Order,
    pub shipping_address: Address,
}
```

## 2. Common Patterns and Anti-patterns

### 2.1 Newtype Pattern for Type Safety
Wrap primitive types in newtype structs to enforce strong typing and prevent logical errors from mixing incompatible IDs or values.

❌ BAD:
```rust
fn process_transaction(user_id: u64, account_id: u64) { /* ... */ }
// Calling with swapped arguments compiles: process_transaction(account.id, user.id);
```

✅ GOOD:
```rust
#[derive(Debug, PartialEq, Eq, Copy, Clone)] pub struct UserId(u64);
#[derive(Debug, PartialEq, Eq, Copy, Clone)] pub struct AccountId(u64);

fn process_transaction(user_id: UserId, account_id: AccountId) { /* ... */ }
// Compiler prevents: process_transaction(account.id, user.id);
```

### 2.2 Builder Pattern for Complex Construction
Use the Builder pattern for structs with many optional fields or complex initialization logic. Avoid it for simple structs with one or two clear constructors.

❌ BAD (for many fields):
```rust
pub struct Config { pub timeout: u64, pub retries: u8 }
impl Config {
    pub fn new(timeout: Option<u64>, retries: Option<u8>) -> Self { /* ... */ }
}
```

✅ GOOD:
```rust
pub struct Config { pub timeout: u64, pub retries: u8 }
impl Config { pub fn builder() -> ConfigBuilder { ConfigBuilder::default() } }

#[derive(Default)] pub struct ConfigBuilder { pub timeout: Option<u64>, pub retries: Option<u8> }
impl ConfigBuilder {
    pub fn timeout(mut self, timeout: u64) -> Self { self.timeout = Some(timeout); self }
    pub fn retries(mut self, retries: u8) -> Self { self.retries = Some(retries); self }
    pub fn build(self) -> Result<Config, &'static str> {
        Ok(Config { timeout: self.timeout.unwrap_or(1000), retries: self.retries.unwrap_or(3) })
    }
}
```

### 2.3 Minimal Generic Bounds on Types
Apply generic bounds only when they are *intrinsic* to the type definition. Place stricter, *derivable* bounds on `impl` blocks or specific functions, not on the struct/enum itself.

❌ BAD:
```rust
pub struct Cache<K: std::hash::Hash + Eq + std::fmt::Debug, V: Clone> { /* ... */ }
```

✅ GOOD:
```rust
pub struct Cache<K, V> { /* ... */ } // K and V are just types

impl<K: std::hash::Hash + Eq, V> Cache<K, V> {
    pub fn new() -> Self { /* ... */ }
}
impl<K: std::fmt::Debug, V: std::fmt::Debug> Cache<K, V> {
    pub fn print_keys(&self) { /* ... */ }
}
```

## 3. Performance Considerations

### 3.1 Default to `Vec` and `HashMap`
Start with `Vec` for sequences and `HashMap` for maps. They are highly optimized and cover most use cases. Only switch to other collections (`VecDeque`, `BTreeMap`, `LinkedList`, etc.) when profiling shows a specific performance bottleneck or a unique functional requirement (e.g., sorted keys for `BTreeMap`).

❌ BAD:
```rust
use std::collections::LinkedList; // Unnecessary for most simple lists
fn process_items(items: LinkedList<String>) { /* ... */ }
```

✅ GOOD:
```rust
use std::collections::HashMap; // Default for key-value pairs
fn process_items(items: Vec<String>) { /* ... */ }
fn lookup_data(data: HashMap<String, u32>) { /* ... */ }
```

### 3.2 Pre-allocate Collection Capacity
When the approximate size of a collection is known beforehand, pre-allocate its capacity to avoid frequent reallocations, which can be costly.

❌ BAD:
```rust
let mut vec = Vec::new();
for i in 0..1000 { vec.push(i); } // Potentially many reallocations
```

✅ GOOD:
```rust
let mut vec = Vec::with_capacity(1000);
for i in 0..1000 { vec.push(i); } // Single allocation
```

## 4. Common Pitfalls and Gotchas

### 4.1 Error Handling: `Result` for Recoverable Errors
Always use `Result<T, E>` for errors that can be handled or propagated. Reserve `panic!` for unrecoverable bugs or situations indicating a programming error.

❌ BAD:
```rust
fn load_config(path: &str) -> String {
    std::fs::read_to_string(path).expect("Failed to read config")
}
```

✅ GOOD:
```rust
fn load_config(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}
```

### 4.2 Document `unsafe` Blocks
Every `unsafe` block *must* be preceded by a `// SAFETY:` comment explaining *why* the code inside is sound and upholds Rust's safety invariants.

❌ BAD:
```rust
let mut vec = vec![0];
let ptr = vec.as_mut_ptr();
unsafe { *ptr = 42; } // Missing safety justification
```

✅ GOOD:
```rust
let mut vec = vec![0];
let ptr = vec.as_mut_ptr();
// SAFETY: `ptr` is a valid pointer to `vec[0]`, and we are writing a valid
// integer `42` to it. The vector has at least one element.
unsafe { *ptr = 42; }
```

## 5. Testing Approaches

### 5.1 Unit Tests with `#[test]`
Use `#[test]` for isolated unit tests, typically placed in a `tests` module within the same file as the code being tested.

```rust
// src/lib.rs
pub fn add(a: i32, b: i32) -> i32 { a + b }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(1, 2), 3);
    }
}
```

### 5.2 Example Tests with `rustdoc`
Leverage `rustdoc` for documentation examples. These examples are compiled and run as tests, ensuring your documentation stays up-to-date and correct.

```rust
/// Adds two integers together.
///
/// # Examples
///
/// ```
/// assert_eq!(my_crate::add(5, 3), 8);
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```