---
description: This rule provides comprehensive best practices for Rust development, covering code organization, common patterns, performance, security, testing, pitfalls, and tooling. It aims to guide developers in writing idiomatic, efficient, secure, and maintainable Rust code.
globs: *.rs
---
# Rust Best Practices

This document outlines a comprehensive set of best practices for Rust development, covering various aspects from code organization to security and tooling. Adhering to these guidelines will help you write idiomatic, efficient, secure, and maintainable Rust code.

## 1. Code Organization and Structure

### 1.1. Directory Structure

-   **`src/`**: Contains all the Rust source code.
    -   **`main.rs`**: The entry point for binary crates.
    -   **`lib.rs`**: The entry point for library crates.
    -   **`bin/`**:  Contains source files for multiple binary executables within the same project.  Each file in `bin/` will be compiled into a separate executable.
    -   **`modules/` or `components/`**: (Optional)  For larger projects, group related modules or components into subdirectories. Use descriptive names.
    -   **`tests/`**:  Integration tests. (See Testing section below for more details.)
    -   **`examples/`**: Example code that demonstrates how to use the library.
-   **`benches/`**: Benchmark tests (using `criterion` or similar).
-   **`Cargo.toml`**: Project manifest file.
-   **`Cargo.lock`**: Records the exact versions of dependencies used. **Do not manually edit.**
-   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
-   **`README.md`**: Project documentation, including usage instructions, build instructions, and license information.
-   **`LICENSE`**: Contains the project's license.


my_project/
├── Cargo.toml
├── Cargo.lock
├── src/
│   ├── main.rs         # Entry point for a binary crate
│   ├── lib.rs          # Entry point for a library crate
│   ├── modules/
│   │   ├── module_a.rs # A module within the crate
│   │   └── module_b.rs # Another module
│   └── bin/
│       ├── cli_tool.rs # A separate binary executable
│       └── worker.rs   # Another binary executable
├── tests/
│   └── integration_test.rs # Integration tests
├── benches/
│   └── my_benchmark.rs # Benchmark tests using Criterion
├── examples/
│   └── example_usage.rs # Example code using the library
├── README.md
└── LICENSE


### 1.2. File Naming Conventions

-   Rust source files use the `.rs` extension.
-   Module files (e.g., `module_a.rs`) should be named after the module they define.
-   Use snake_case for file names (e.g., `my_module.rs`).

### 1.3. Module Organization

-   Use modules to organize code into logical units.
-   Declare modules in `lib.rs` or `main.rs` using the `mod` keyword.
-   Use `pub mod` to make modules public.
-   Create separate files for each module to improve readability and maintainability.
-   Use `use` statements to bring items from other modules into scope.

rust
// lib.rs

pub mod my_module;

mod internal_module; // Not public


rust
// my_module.rs

pub fn my_function() {
    //...
}


### 1.4. Component Architecture

-   For larger applications, consider using a component-based architecture.
-   Each component should be responsible for a specific part of the application's functionality.
-   Components should communicate with each other through well-defined interfaces (traits).
-   Consider using dependency injection to decouple components and improve testability.

### 1.5. Code Splitting Strategies

-   Split code into smaller, reusable modules.
-   Use feature flags to conditionally compile code for different platforms or features.
-   Consider using dynamic linking (if supported by your target platform) to reduce binary size.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **Builder Pattern**: For constructing complex objects with many optional parameters.
-   **Factory Pattern**: For creating objects without specifying their concrete types.
-   **Observer Pattern**: For implementing event-driven systems.
-   **Strategy Pattern**: For selecting algorithms at runtime.
-   **Visitor Pattern**: For adding new operations to existing data structures without modifying them.

### 2.2. Recommended Approaches for Common Tasks

-   **Data Structures**: Use `Vec` for dynamic arrays, `HashMap` for key-value pairs, `HashSet` for unique elements, `BTreeMap` and `BTreeSet` for sorted collections.
-   **Concurrency**: Use `Arc` and `Mutex` for shared mutable state, channels for message passing, and the `rayon` crate for data parallelism.
-   **Asynchronous Programming**: Use `async` and `await` for writing asynchronous code.
-   **Error Handling**: Use the `Result` type for recoverable errors and `panic!` for unrecoverable errors.

### 2.3. Anti-patterns and Code Smells

-   **Unnecessary Cloning**: Avoid cloning data unless it is absolutely necessary. Use references instead.
-   **Excessive `unwrap()` Calls**: Handle errors properly instead of using `unwrap()`, which can cause the program to panic.
-   **Overuse of `unsafe`**: Minimize the use of `unsafe` code and carefully review any unsafe code to ensure it is correct.
-   **Ignoring Compiler Warnings**: Treat compiler warnings as errors and fix them.
-   **Premature Optimization**: Focus on writing clear, correct code first, and then optimize only if necessary.

### 2.4. State Management

-   **Immutability by Default**: Prefer immutable data structures and functions that return new values instead of modifying existing ones.
-   **Ownership and Borrowing**: Use Rust's ownership and borrowing system to manage memory and prevent data races.
-   **Interior Mutability**: Use `Cell`, `RefCell`, `Mutex`, and `RwLock` for interior mutability when necessary, but be careful to avoid data races.

### 2.5. Error Handling

-   **`Result<T, E>`**: Use `Result` to represent fallible operations. `T` is the success type, and `E` is the error type.
-   **`Option<T>`**: Use `Option` to represent the possibility of a missing value. `Some(T)` for a value, `None` for no value.
-   **`?` Operator**: Use the `?` operator to propagate errors up the call stack.
-   **Custom Error Types**: Define custom error types using enums or structs to provide more context about errors.
-   **`anyhow` and `thiserror` Crates**: Consider using the `anyhow` crate for simple error handling and the `thiserror` crate for defining custom error types.

## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Profiling**: Use profiling tools (e.g., `perf`, `cargo flamegraph`) to identify performance bottlenecks.
-   **Benchmarking**: Use benchmarking tools (e.g., `criterion`) to measure the performance of code changes.
-   **Zero-Cost Abstractions**: Leverage Rust's zero-cost abstractions, such as iterators, closures, and generics.
-   **Inlining**: Use the `#[inline]` attribute to encourage the compiler to inline functions.
-   **LTO (Link-Time Optimization)**: Enable LTO to improve performance by optimizing across crate boundaries.

### 3.2. Memory Management

-   **Minimize Allocations**: Reduce the number of allocations and deallocations by reusing memory and using stack allocation when possible.
-   **Avoid Copying Large Data Structures**: Use references or smart pointers to avoid copying large data structures.
-   **Use Efficient Data Structures**: Choose the right data structure for the job based on its performance characteristics.
-   **Consider `Box` and `Rc`**: `Box` for single ownership heap allocation, `Rc` and `Arc` for shared ownership (latter thread-safe).

### 3.3. Rendering Optimization

-   **(Relevant if the Rust application involves rendering, e.g., a game or GUI)**
-   **Batch draw calls**: Combine multiple draw calls into a single draw call to reduce overhead.
-   **Use efficient data structures**: Use data structures that are optimized for rendering, such as vertex buffers and index buffers.
-   **Profile rendering performance**: Use profiling tools to identify rendering bottlenecks.

### 3.4. Bundle Size Optimization

-   **Strip Debug Symbols**: Remove debug symbols from release builds to reduce binary size.
-   **Enable LTO**: LTO can also reduce binary size by removing dead code.
-   **Use `minisize` Profile**: Create a `minisize` profile in `Cargo.toml` for optimizing for size.
-   **Avoid Unnecessary Dependencies**: Only include the dependencies that are absolutely necessary.

### 3.5. Lazy Loading

-   **Load Resources on Demand**: Load resources (e.g., images, sounds, data files) only when they are needed.
-   **Use a Loading Screen**: Display a loading screen while resources are being loaded.
-   **Consider Streaming**: Stream large resources from disk or network instead of loading them all at once.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

-   **Buffer Overflows**: Prevent buffer overflows by using safe indexing methods (e.g., `get()`, `get_mut()`) and validating input sizes.
-   **SQL Injection**: Prevent SQL injection by using parameterized queries and escaping user input.
-   **Cross-Site Scripting (XSS)**: Prevent XSS by escaping user input when rendering HTML.
-   **Command Injection**: Prevent command injection by avoiding the use of `std::process::Command` with user-supplied arguments.
-   **Denial of Service (DoS)**: Protect against DoS attacks by limiting resource usage (e.g., memory, CPU, network connections).
-   **Integer Overflows**:  Use the `checked_add`, `checked_sub`, `checked_mul`, etc. methods on integers to prevent overflows.
-   **Use-After-Free**:  Rust's ownership system largely prevents this, but be cautious when using `unsafe` code or dealing with raw pointers.
-   **Data Races**:  Avoid data races by using appropriate synchronization primitives (`Mutex`, `RwLock`, channels).
-   **Uninitialized Memory**: Rust generally initializes memory, but `unsafe` code can bypass this.  Be careful when working with uninitialized memory.

### 4.2. Input Validation

-   **Validate All Input**: Validate all input from external sources, including user input, network data, and file contents.
-   **Use a Whitelist Approach**: Define a set of allowed values and reject any input that does not match.
-   **Sanitize Input**: Remove or escape any potentially dangerous characters from input.
-   **Limit Input Length**: Limit the length of input strings to prevent buffer overflows.
-   **Check Data Types**: Ensure that input data is of the expected type.

### 4.3. Authentication and Authorization

-   **Use Strong Passwords**: Require users to create strong passwords and store them securely using a hashing algorithm like Argon2 or bcrypt.
-   **Implement Two-Factor Authentication (2FA)**: Add an extra layer of security by requiring users to authenticate with a second factor, such as a code from their phone.
-   **Use JSON Web Tokens (JWT)**: Use JWTs for stateless authentication and authorization.
-   **Implement Role-Based Access Control (RBAC)**: Define roles with specific permissions and assign users to those roles.
-   **Principle of Least Privilege**:  Grant users only the minimum necessary privileges to perform their tasks.
-   **Regular Audits**: Perform regular security audits of authentication and authorization mechanisms.

### 4.4. Data Protection

-   **Encrypt Sensitive Data**: Encrypt sensitive data at rest and in transit using strong encryption algorithms like AES-256.
-   **Use HTTPS**: Use HTTPS to encrypt communication between the client and the server.
-   **Protect API Keys**: Store API keys securely and restrict their usage to authorized users.
-   **Handle Secrets Securely**: Use environment variables or dedicated secret management tools (e.g., Vault, AWS Secrets Manager) to store secrets.
-   **Avoid Hardcoding Secrets**: Never hardcode secrets directly into the code.
-   **Data Masking/Redaction**: Mask or redact sensitive data when logging or displaying it.

### 4.5. Secure API Communication

-   **Use TLS/SSL**: Enforce TLS/SSL for all API communication.
-   **Validate Certificates**: Properly validate server certificates to prevent man-in-the-middle attacks.
-   **Rate Limiting**: Implement rate limiting to prevent abuse and DoS attacks.
-   **API Versioning**: Use API versioning to maintain backward compatibility and allow for future changes.
-   **Input and Output Validation**: Thoroughly validate both input to and output from the API.
-   **Content Security Policy (CSP)**: Use CSP headers to prevent XSS attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

-   **Test Individual Units of Code**: Write unit tests to verify the correctness of individual functions, modules, and components.
-   **Use the `#[test]` Attribute**: Use the `#[test]` attribute to mark functions as unit tests.
-   **Use `assert!` and `assert_eq!`**: Use `assert!` and `assert_eq!` macros to check that the code behaves as expected.
-   **Test Driven Development (TDD)**: Consider writing tests before writing code.
-   **Table-Driven Tests**:  Use parameterized tests or table-driven tests for testing multiple scenarios with different inputs.

rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }
}


### 5.2. Integration Testing

-   **Test Interactions Between Components**: Write integration tests to verify that different components of the application work together correctly.
-   **Create a `tests/` Directory**: Place integration tests in a `tests/` directory at the root of the project.
-   **Use Separate Test Files**: Create separate test files for each integration test.

### 5.3. End-to-End Testing

-   **Test the Entire Application**: Write end-to-end tests to verify that the entire application works as expected.
-   **Use a Testing Framework**: Use a testing framework (e.g., `cucumber`, `selenium`) to automate end-to-end tests.
-   **Test User Flows**: Test common user flows to ensure that the application is usable.

### 5.4. Test Organization

-   **Group Tests by Functionality**: Organize tests into modules and submodules based on the functionality they test.
-   **Use Descriptive Test Names**: Use descriptive test names that clearly indicate what the test is verifying.
-   **Keep Tests Separate from Production Code**: Keep tests in separate files and directories to avoid cluttering the production code.
-   **Run tests frequently**: Integrate tests into your development workflow and run them frequently to catch errors early.

### 5.5. Mocking and Stubbing

-   **Use Mocking Libraries**: Use mocking libraries (e.g., `mockall`, `mockito`) to create mock objects for testing.
-   **Use Traits for Interfaces**: Define traits for interfaces to enable mocking and stubbing.
-   **Avoid Global State**: Avoid global state to make it easier to mock and stub dependencies.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Borrowing Rules**: Misunderstanding Rust's borrowing rules can lead to compile-time errors. Ensure you understand ownership, borrowing, and lifetimes.
-   **Move Semantics**: Be aware of move semantics and how they affect ownership. Data is moved by default, not copied.
-   **Lifetime Annotations**: Forgetting lifetime annotations can lead to compile-time errors. Annotate lifetimes when necessary.
-   **Error Handling**: Not handling errors properly can lead to unexpected panics. Use `Result` and the `?` operator to handle errors gracefully.
-   **Unsafe Code**: Overusing or misusing `unsafe` code can lead to undefined behavior and security vulnerabilities.

### 6.2. Edge Cases

-   **Integer Overflow**: Be aware of integer overflow and use checked arithmetic methods to prevent it.
-   **Unicode**: Handle Unicode characters correctly to avoid unexpected behavior.
-   **File Paths**: Handle file paths correctly, especially when dealing with different operating systems.
-   **Concurrency**: Be careful when writing concurrent code to avoid data races and deadlocks.

### 6.3. Version-Specific Issues

-   **Check Release Notes**: Review the release notes for new versions of Rust to identify any breaking changes or new features that may affect your code.
-   **Use `rustup`**: Use `rustup` to manage multiple versions of Rust.
-   **Update Dependencies**: Keep your dependencies up to date to take advantage of bug fixes and new features.

### 6.4. Compatibility Concerns

-   **C Interoperability**: Be careful when interacting with C code to avoid undefined behavior.
-   **Platform-Specific Code**: Use conditional compilation to handle platform-specific code.
-   **WebAssembly**: Be aware of the limitations of WebAssembly when targeting the web.

### 6.5. Debugging Strategies

-   **Use `println!`**: Use `println!` statements to print debugging information.
-   **Use a Debugger**: Use a debugger (e.g., `gdb`, `lldb`) to step through the code and inspect variables.
-   **Use `assert!`**: Use `assert!` to check that the code behaves as expected.
-   **Use Logging**: Use a logging library (e.g., `log`, `tracing`) to record debugging information.
-   **Clippy**: Use Clippy to catch common mistakes and improve code quality.
-   **cargo-flamegraph**: Use cargo-flamegraph to profile and visualize the execution of your code.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **Rustup**: For managing Rust toolchains and versions.
-   **Cargo**: The Rust package manager and build tool.
-   **IDE/Editor**: VS Code with the rust-analyzer extension, IntelliJ Rust, or other editors with Rust support.
-   **Clippy**: A linter for Rust code.
-   **Rustfmt**: A code formatter for Rust code.
-   **Cargo-edit**: A utility for easily modifying `Cargo.toml` dependencies.
-   **Cargo-watch**: Automatically runs tests on file changes.
-   **lldb or GDB**: Debuggers for Rust applications.

### 7.2. Build Configuration

-   **Use `Cargo.toml`**: Configure build settings, dependencies, and metadata in the `Cargo.toml` file.
-   **Use Profiles**: Define different build profiles for development, release, and testing.
-   **Feature Flags**: Use feature flags to conditionally compile code for different platforms or features.

toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
rand = "0.8"

[features]
default = ["serde"] # 'default' feature enables 'serde'
expensive_feature = []

[profile.release]
opt-level = 3
debug = false
lto = true


### 7.3. Linting and Formatting

-   **Use Clippy**: Use Clippy to catch common mistakes and enforce coding standards.
-   **Use Rustfmt**: Use Rustfmt to automatically format code according to the Rust style guide.
-   **Configure Editor**: Configure your editor to automatically run Clippy and Rustfmt on save.
-   **Pre-commit Hooks**: Set up pre-commit hooks to run Clippy and Rustfmt before committing code.

shell
# Install Clippy
rustup component add clippy

# Run Clippy
cargo clippy

# Install Rustfmt
rustup component add rustfmt

# Run Rustfmt
cargo fmt


### 7.4. Deployment Best Practices

-   **Build Release Binaries**:  Build your application in release mode (`cargo build --release`) to optimize for performance.
-   **Minimize Dependencies**: Reduce the number of dependencies to minimize the size of the deployed application.
-   **Containerization (Docker)**: Use Docker to create a consistent and reproducible deployment environment.
-   **Static Linking**: Consider static linking to create a single executable file.
-   **Process Manager (systemd, supervisord)**:  Use a process manager to ensure your application restarts automatically if it crashes.

### 7.5. CI/CD Integration

-   **Use a CI/CD System**: Use a CI/CD system (e.g., GitHub Actions, GitLab CI, Jenkins) to automate the build, test, and deployment process.
-   **Run Tests on CI**: Run unit tests, integration tests, and end-to-end tests on CI.
-   **Run Linters and Formatters on CI**: Run Clippy and Rustfmt on CI to enforce coding standards.
-   **Automate Deployment**: Automate the deployment process to reduce manual effort and errors.


# Example GitHub Actions workflow
name: CI

on:
  push:
    branches:
      - main
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
      - uses: actions/cache@v3
        with:
          path: | 
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Install Rust
        run: rustup default stable

      - name: Build
        run: cargo build --verbose

      - name: Run tests
        run: cargo test --verbose

      - name: Run clippy
        run: cargo clippy -- -D warnings


By following these best practices, you can write high-quality Rust code that is efficient, secure, and maintainable. Remember to stay up-to-date with the latest Rust features and best practices to continuously improve your skills and knowledge.