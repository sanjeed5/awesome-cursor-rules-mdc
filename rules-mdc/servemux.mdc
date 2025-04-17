---
description: This rule enforces best practices for using the `net/http` ServeMux in Go, promoting clean, maintainable, and efficient code. It covers routing, handler design, and error handling specifics to help developers leverage ServeMux effectively.
globs: *.go
---
---
# servemux Best Practices and Coding Standards

This document provides comprehensive guidelines for using the `net/http` ServeMux in Go, promoting best practices, coding standards, and efficient development.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices

-   **Modular Design:** Organize your code into logical modules, each responsible for a specific set of functionalities.
-   **Handler Grouping:** Group related handlers within the same directory or package. For instance, all user-related handlers could reside in a `users` package.
-   **Middleware Directory:** Create a dedicated `middleware` directory for reusable middleware functions.
-   **Internal vs. External:** Utilize the `internal` directory to encapsulate code that should not be exposed outside your module.  This enhances encapsulation and reduces the API surface.


project-root/
├── cmd/
│   └── my-app/
│       └── main.go
├── internal/
│   ├── handlers/
│   │   ├── users.go
│   │   └── products.go
│   └── middleware/
│       ├── auth.go
│       └── logging.go
├── pkg/
│   └── utils/
│       └── utils.go
└── go.mod


### 1.2 File Naming Conventions

-   **Descriptive Names:** Use clear and descriptive file names that reflect the functionality they contain. For example, `users.go` for user-related handlers, and `auth.go` for authentication middleware.
-   **Handler Specifics:** If a file contains a specific handler, name it accordingly. For example, `get_user_handler.go` or `create_product_handler.go`.
-   **Lowercase:** Use lowercase for all file names.

### 1.3 Module Organization

-   **`go.mod`:** Ensure your project has a `go.mod` file to manage dependencies.  Run `go mod init <module_name>` to create one.
-   **Semantic Versioning:** Follow semantic versioning for your modules.  Use tags (e.g., `v1.0.0`) to mark releases.
-   **Vendor Dependencies:** Consider vendoring dependencies using `go mod vendor` to ensure reproducibility.

### 1.4 Component Architecture

-   **Separation of Concerns:** Design your components to have a single responsibility.  Separate handler logic from business logic and data access.
-   **Interfaces:** Use interfaces to define contracts between components, promoting loose coupling and testability. For example:

    go
type UserStore interface {
    GetUser(id int) (*User, error)
    CreateUser(user *User) error
}

type UserHandler struct {
    store UserStore
}
    

### 1.5 Code Splitting Strategies

-   **Package-Based Splitting:** Divide your application into packages based on functionality.  Each package should be cohesive and have a clear purpose.
-   **Feature-Based Splitting:**  Organize code based on features.  Each feature gets its own directory or package containing all relevant components.
-   **Layered Architecture:**  Implement a layered architecture (e.g., presentation, business logic, data access) and split the code accordingly.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to servemux

-   **Middleware Chaining:** Implement middleware as a chain of functions that decorate the handler. This allows for modular and reusable logic. Example:

    go
func LoggingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        log.Printf("Request: %s %s", r.Method, r.URL.Path)
        next.ServeHTTP(w, r)
    })
}

// Usage:
mux := http.NewServeMux()
mux.Handle("/", LoggingMiddleware(http.HandlerFunc(myHandler)))
    

-   **Context-Aware Handlers:** Utilize the `context.Context` to pass request-scoped values to handlers. This facilitates tracing, request cancellation, and data sharing.

    go
func MyHandler(w http.ResponseWriter, r *http.Request) {
    userID := r.Context().Value("userID").(int)
    // ...
}

// Setting context value in middleware:
ctx := context.WithValue(r.Context(), "userID", 123)
r = r.WithContext(ctx)
    

-   **Route Grouping (with custom muxes):** Use separate ServeMux instances for different route groups (e.g., API v1, API v2, admin routes).  This improves organization and maintainability.

    go
v1Mux := http.NewServeMux()
v1Mux.HandleFunc("/users", v1UsersHandler)
v1Mux.HandleFunc("/products", v1ProductsHandler)

v2Mux := http.NewServeMux()
v2Mux.HandleFunc("/users", v2UsersHandler)
v2Mux.HandleFunc("/products", v2ProductsHandler)

http.Handle("/api/v1/", v1Mux)
http.Handle("/api/v2/", v2Mux)
    

### 2.2 Recommended Approaches for Common Tasks

-   **Creating a New ServeMux:** Always use `http.NewServeMux()` to create a new ServeMux instance. Avoid using the default `http.DefaultServeMux` to prevent global state issues. Example:

    go
mux := http.NewServeMux()
mux.HandleFunc("/", myHandler)
http.ListenAndServe(":8080", mux)
    

-   **Registering Handlers:**  Use `mux.HandleFunc()` or `mux.Handle()` to register handlers with the ServeMux.

    go
mux.HandleFunc("/users", usersHandler)
mux.Handle("/products", productHandler{})
    

-   **Serving Static Files:** Use `http.FileServer()` to serve static files.

    go
fs := http.FileServer(http.Dir("./static"))
mux.Handle("/static/", http.StripPrefix("/static/", fs))
    

-   **Handling HTTP Methods:**  Use `http.MethodGet`, `http.MethodPost`, etc., constants to check the HTTP method.  With Go 1.22+, specify methods in the pattern itself for `HandleFunc`.

    go
mux.HandleFunc("GET /users/{id}", getUserHandler)
mux.HandleFunc("POST /users", createUserHandler)
    

### 2.3 Anti-patterns and Code Smells to Avoid

-   **Global State:** Avoid using global variables or the default `http.DefaultServeMux` for request handling, as it can lead to race conditions and difficult debugging.
-   **Overlapping Routes:** Be mindful of overlapping routes.  Go 1.22+ will panic if conflicting patterns are registered. Ensure that your routes are distinct and do not cause conflicts. Properly consider precedence rules.
-   **Ignoring Errors:**  Always handle errors returned by functions. Log errors and return appropriate HTTP status codes to the client.
-   **Long Handler Functions:**  Keep handler functions short and focused. Delegate complex logic to other functions or services.
-   **Hardcoding Values:**  Avoid hardcoding configuration values.  Use environment variables or configuration files.
-   **Lack of Input Validation:** Always validate user input to prevent security vulnerabilities.

### 2.4 State Management Best Practices

-   **Stateless Handlers:** Design handlers to be stateless whenever possible. If state is required, store it in the request context or an external data store.
-   **Request-Scoped Values:**  Use the `context.Context` to store request-scoped values. Avoid using global variables to store state related to a specific request.
-   **Sessions:** Use secure session management techniques to maintain user sessions.

### 2.5 Error Handling Patterns

-   **Centralized Error Handling:**  Implement a centralized error handling middleware to catch and log errors. Return appropriate HTTP status codes and error messages to the client.

    go
func ErrorHandlingMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if err := recover(); err != nil {
                log.Printf("Panic: %v", err)
                http.Error(w, "Internal Server Error", http.StatusInternalServerError)
            }
        }()
        next.ServeHTTP(w, r)
    })
}
    

-   **Custom Error Types:** Define custom error types to provide more context about the error.

    go
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("Validation error: %s - %s", e.Field, e.Message)
}
    

-   **Logging Errors:** Log errors with sufficient detail for debugging.
-   **Returning Appropriate Status Codes:** Return appropriate HTTP status codes based on the error. Use descriptive error messages in the response body.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Connection Pooling:** Use connection pooling for database connections to reduce overhead.
-   **Caching:** Implement caching for frequently accessed data to reduce database load.
-   **Gzip Compression:** Enable Gzip compression for responses to reduce bandwidth usage.

    go
func GzipMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
            gz, err := gzip.NewWriterLevel(w, gzip.BestSpeed)
            if err != nil {
                http.Error(w, err.Error(), http.StatusInternalServerError)
                return
            }
            defer gz.Close()
            w.Header().Set("Content-Encoding", "gzip")
            next.ServeHTTP(gzipResponseWriter{Writer: gz, ResponseWriter: w}, r)
        } else {
            next.ServeHTTP(w, r)
        }
    })
}

type gzipResponseWriter struct {
    io.Writer
    http.ResponseWriter
}

func (w gzipResponseWriter) Write(b []byte) (int, error) {
    return w.Writer.Write(b)
}
    

-   **Keep-Alive Connections:** Enable keep-alive connections to reduce connection establishment overhead.
-   **Efficient Data Structures:**  Use efficient data structures and algorithms for data processing.

### 3.2 Memory Management

-   **Avoid Memory Leaks:** Be aware of potential memory leaks, especially when using goroutines. Ensure that all goroutines are properly terminated.
-   **Object Pooling:** Consider using object pooling for frequently allocated objects.
-   **String Conversions:**  Minimize unnecessary string conversions, as they can be expensive.
-   **Defer Statements:**  Use `defer` statements carefully. While convenient, they can add overhead if overused in performance-critical sections.

### 3.3 Rendering Optimization

-   **Template Caching:** Cache templates to reduce parsing overhead.
-   **Efficient Template Rendering:** Use efficient template rendering techniques. Consider using pre-compiled templates.
-   **Minimize DOM Updates:** Minimize DOM updates in the client-side JavaScript code.

### 3.4 Bundle Size Optimization

-   **Code Minification:** Minify JavaScript and CSS code to reduce bundle size.
-   **Tree Shaking:** Use tree shaking to remove unused code from the bundle.
-   **Image Optimization:** Optimize images to reduce file size.

### 3.5 Lazy Loading Strategies

-   **Lazy Loading Images:** Lazy load images that are not immediately visible.
-   **Code Splitting:** Split the code into smaller chunks that can be loaded on demand.
-   **Dynamic Imports:** Use dynamic imports to load modules only when they are needed.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities and How to Prevent Them

-   **SQL Injection:** Prevent SQL injection by using parameterized queries or ORMs.
-   **Cross-Site Scripting (XSS):** Prevent XSS by escaping user input before rendering it in HTML.
-   **Cross-Site Request Forgery (CSRF):** Prevent CSRF by using CSRF tokens.
-   **Authentication Bypass:** Implement robust authentication and authorization mechanisms to prevent authentication bypass.
-   **Denial-of-Service (DoS):** Implement rate limiting and request validation to prevent DoS attacks.

### 4.2 Input Validation

-   **Validate All Inputs:** Validate all user inputs to ensure that they conform to the expected format and values. Use libraries like `ozzo-validation` for complex validation rules.
-   **Sanitize Inputs:** Sanitize inputs to remove potentially malicious characters.
-   **Whitelist Inputs:** Use a whitelist approach to only allow specific characters or patterns.

### 4.3 Authentication and Authorization Patterns

-   **Authentication Middleware:** Implement authentication middleware to verify user credentials.
-   **Authorization Middleware:** Implement authorization middleware to check user permissions.
-   **JWT (JSON Web Tokens):** Use JWT for stateless authentication.
-   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization.
-   **Role-Based Access Control (RBAC):** Implement RBAC to manage user permissions.

### 4.4 Data Protection Strategies

-   **Encryption:** Encrypt sensitive data at rest and in transit.
-   **Hashing:** Hash passwords with a strong hashing algorithm like bcrypt.
-   **Salting:** Use salts to prevent rainbow table attacks.
-   **Data Masking:** Mask sensitive data in logs and error messages.

### 4.5 Secure API Communication

-   **HTTPS:** Use HTTPS to encrypt communication between the client and the server.
-   **TLS (Transport Layer Security):** Use TLS 1.3 or higher for secure communication.
-   **Certificate Pinning:** Consider using certificate pinning to prevent man-in-the-middle attacks.
-   **Rate Limiting:** Implement rate limiting to prevent abuse.

## 5. Testing Approaches

### 5.1 Unit Testing Strategies

-   **Test Individual Units:** Unit test individual functions and methods in isolation.
-   **Mock Dependencies:** Use mocks to isolate the unit under test from its dependencies. Use libraries like `gomock` or `testify/mock`.
-   **Test Boundary Conditions:** Test boundary conditions and edge cases.
-   **Table-Driven Tests:** Use table-driven tests to test multiple scenarios with different inputs and outputs.

### 5.2 Integration Testing

-   **Test Interactions Between Components:** Integration test the interactions between different components of the application.
-   **Test Database Interactions:** Test the interactions with the database.
-   **Test API Endpoints:** Test the API endpoints.

### 5.3 End-to-End Testing

-   **Test the Entire Application:** End-to-end test the entire application flow.
-   **Use Automated Testing Tools:** Use automated testing tools like Selenium or Cypress.

### 5.4 Test Organization

-   **Keep Tests Close to Code:** Keep test files in the same directory as the code they test. Use the `_test.go` suffix for test files.
-   **Separate Test Packages:** Consider creating separate test packages for integration and end-to-end tests.
-   **Descriptive Test Names:** Use descriptive test names that clearly indicate what is being tested.

### 5.5 Mocking and Stubbing

-   **Use Interfaces:** Define interfaces for dependencies to facilitate mocking.
-   **Use Mocking Libraries:** Use mocking libraries like `gomock` or `testify/mock` to generate mocks.
-   **Create Stub Implementations:** Create stub implementations for dependencies that are difficult to mock.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes Developers Make

-   **Forgetting to Handle Errors:** Always check and handle errors returned by functions.
-   **Ignoring Context Cancellation:** Respect context cancellation signals and terminate long-running operations.
-   **Not Validating Inputs:** Always validate user inputs to prevent security vulnerabilities.
-   **Using Global State:** Avoid using global variables for request handling, as it can lead to race conditions.
-   **Overlapping Routes:**  Failing to understand route precedence, especially in Go 1.22+ can lead to unexpected behavior or panics during server initialization. Thoroughly test route registrations.

### 6.2 Edge Cases to Be Aware Of

-   **Trailing Slashes:** Be aware of how trailing slashes affect route matching. In Go 1.22+, use `{$}` to match only the path with the trailing slash.
-   **Wildcard Matching:** Understand how wildcards match different segments of the path. Use `{pathname...}` to match all remaining segments.
-   **Method Matching:** Ensure that the correct HTTP method is being used for each route. With Go 1.22+, be aware that `GET` also matches `HEAD` requests.

### 6.3 Version-Specific Issues

-   **Go 1.22 Routing Enhancements:** Be aware of the new routing enhancements in Go 1.22, including method matching and wildcard support. Understand the new precedence rules.
-   **Compatibility Issues:** Check for compatibility issues when upgrading to new versions of Go or third-party libraries.

### 6.4 Compatibility Concerns

-   **Backwards Compatibility:** Ensure that changes do not break backwards compatibility. Use feature flags to gradually introduce new features.
-   **API Versioning:** Use API versioning to manage changes to the API.
-   **Client Compatibility:** Ensure that client applications are compatible with the API.

### 6.5 Debugging Strategies

-   **Logging:** Use logging to track the execution flow and identify errors.
-   **Debugging Tools:** Use debugging tools like `delve` to step through the code and inspect variables.
-   **Profiling:** Use profiling tools to identify performance bottlenecks.
-   **Tracing:** Use tracing tools to track requests across different services.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

-   **GoLand:** A powerful IDE for Go development.
-   **Visual Studio Code with Go Extension:** A lightweight and versatile code editor with excellent Go support.
-   **Delve:** A Go debugger.
-   **Profiling Tools:** `pprof` is the built-in Go profiler.

### 7.2 Build Configuration

-   **Makefile:** Use a Makefile to automate common build tasks.
-   **`go build`:** Use `go build` to compile the application.
-   **`go test`:** Use `go test` to run tests.

### 7.3 Linting and Formatting

-   **`go fmt`:** Use `go fmt` to format the code according to the Go style guidelines.
-   **`go vet`:** Use `go vet` to identify potential errors in the code.
-   **`golangci-lint`:** Use `golangci-lint` for more advanced linting.

### 7.4 Deployment Best Practices

-   **Containerization:** Use containers (e.g., Docker) to package the application and its dependencies.
-   **Orchestration:** Use orchestration tools (e.g., Kubernetes) to manage and scale the application.
-   **Immutable Infrastructure:** Deploy immutable infrastructure to ensure consistency and reproducibility.

### 7.5 CI/CD Integration

-   **Continuous Integration:** Integrate with a CI/CD pipeline to automate the build, test, and deployment process.
-   **Automated Testing:** Run automated tests as part of the CI/CD pipeline.
-   **Automated Deployment:** Automate the deployment process using tools like Jenkins, CircleCI, or GitHub Actions.

## Additional Best Practices

-   **Meaningful Handler Names:** Use meaningful, concise names for handlers, avoiding repetition of context in function names (as per the Google Style Guide).
-   **Document Code:** Write clear and concise documentation for all functions, methods, and types.
-   **Code Reviews:** Conduct regular code reviews to ensure code quality and identify potential issues.
-   **Stay Updated:** Stay up-to-date with the latest Go best practices and security vulnerabilities.

By following these best practices and coding standards, you can develop robust, maintainable, and efficient applications using the `net/http` ServeMux in Go.