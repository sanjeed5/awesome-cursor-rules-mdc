---
description: Definitive guidelines for writing clear, simple, performant, and maintainable Go code, adhering to Google's style and modern best practices.
globs: **/*
---
# Go Best Practices

This guide outlines our team's definitive Go coding standards, emphasizing clarity, simplicity, performance, and maintainability. Adhere to these principles to ensure a consistent, high-quality codebase that aligns with Google's Go Style Guide.

## 1. Code Organization & Structure

**Prioritize clarity and discoverability.** Structure your projects logically, making it easy for new team members to understand the codebase.

### Package Naming & Purpose
Packages must have short, lowercase names that clearly describe their single purpose. Avoid generic names like `util` or `common`.

❌ BAD:
```go
// myproject/utils/string_helpers.go
package utils // Too generic, doesn't describe specific utility

func SanitizeString(s string) string { /* ... */ }
```

✅ GOOD:
```go
// myproject/text/sanitizer.go
package text // Clear domain, specific functionality

func Sanitize(s string) string { /* ... */ }
```

### `internal` Package for Encapsulation
Always use the `internal` directory for code that should not be imported by other projects or external modules. This enforces strong encapsulation and prevents unintended dependencies.

❌ BAD:
```go
// myproject/service/auth.go (Exposes internal logic to external consumers)
package service

type AuthService struct { /* ... */ }
```

✅ GOOD:
```go
// myproject/internal/auth/service.go (Only importable by modules within 'myproject')
package auth

type Service struct { /* ... */ }
```

## 2. Error Handling

**Handle errors explicitly and provide actionable context.** Never ignore errors. Use `errors.Is` and `errors.As` for robust error inspection.

### Wrap Errors with Context
Always wrap errors to add context, making debugging easier and preserving the error chain. Use `%w` with `fmt.Errorf` to enable `errors.Is` and `errors.As`.

❌ BAD:
```go
if err := db.Save(user); err != nil {
    return fmt.Errorf("failed to save user: %v", err) // Loses original error type
}
```

✅ GOOD:
```go
if err := db.Save(user); err != nil {
    return fmt.Errorf("failed to save user %s: %w", user.ID, err) // Original error preserved
}
```

### Use `errors.Is` and `errors.As` for Inspection
For checking specific error types or values, use `errors.Is` for direct comparison and `errors.As` for unwrapping to a specific error type. This works correctly with wrapped errors.

❌ BAD:
```go
if err == sql.ErrNoRows { // Fails if error is wrapped by fmt.Errorf
    return ErrNotFound
}
if _, ok := err.(*MyCustomError); ok { // Fails if error is wrapped
    // ...
}
```

✅ GOOD:
```go
if errors.Is(err, sql.ErrNoRows) { // Correctly identifies sql.ErrNoRows in chain
    return ErrNotFound
}
var customErr *MyCustomError
if errors.As(err, &customErr) { // Correctly unwraps to MyCustomError
    // ...
}
```

## 3. Performance Considerations

**Minimize allocations and GC pressure.** Go's GC is efficient, but excessive allocations will always be a bottleneck in hot paths.

### Pre-allocate Slices and Maps
Always pre-allocate slices and maps with known or estimated capacities to avoid repeated reallocations and garbage generation.

❌ BAD:
```go
var results []Result // Grows dynamically, causing multiple reallocations
for _, item := range items {
    results = append(results, process(item))
}
```

✅ GOOD:
```go
results := make([]Result, 0, len(items)) // Pre-allocate capacity
for _, item := range items {
    results = append(results, process(item))
}
```

### Use `strings.Builder` for String Concatenation
Avoid `+` or `fmt.Sprintf` in loops for building strings. `strings.Builder` minimizes allocations by writing to a single underlying buffer.

❌ BAD:
```go
var s string
for i := 0; i < 1000; i++ {
    s += strconv.Itoa(i) + "," // Creates many intermediate string allocations
}
```

✅ GOOD:
```go
var sb strings.Builder
sb.Grow(1000 * 5) // Estimate required size to avoid initial reallocations
for i := 0; i < 1000; i++ {
    sb.WriteString(strconv.Itoa(i))
    sb.WriteString(",")
}
s := sb.String()
```

### Leverage `sync.Pool` for Reusable Objects
For frequently created and discarded objects (e.g., HTTP request buffers, database connections), use `sync.Pool` to reduce GC overhead by reusing objects.

❌ BAD:
```go
func handleRequest(w http.ResponseWriter, r *http.Request) {
    buf := make([]byte, 4096) // New allocation per request
    // ... use buf ...
}
```

✅ GOOD:
```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096) // Allocate new buffer if pool is empty
    },
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
    buf := bufferPool.Get().([]byte) // Get a buffer from the pool
    defer bufferPool.Put(buf)        // Return buffer to pool when done
    // ... use buf ...
}
```

### Pass Small Structs by Value
For small, immutable structs (e.g., coordinates, simple IDs), passing by value can avoid heap allocations and improve cache locality, as determined by escape analysis.

❌ BAD:
```go
type Point struct { X, Y int }
func movePoint(p *Point, dx, dy int) { // Pointer might force heap allocation
    p.X += dx
    p.Y += dy
}
```

✅ GOOD:
```go
type Point struct { X, Y int }
func movePoint(p Point, dx, dy int) Point { // Value type, likely stack allocated
    p.X += dx
    p.Y += dy
    return p
}
```

## 4. Concurrency & Context

**Manage goroutines and resource lifetimes effectively.** Always use `context.Context` for cancellation and timeouts.

### Use `context.Context` for Cancellation and Timeouts
Pass `context.Context` as the first argument to functions that perform I/O or long-running operations. This enables graceful shutdown and resource management.

❌ BAD:
```go
func fetchData(url string) ([]byte, error) {
    // No way to cancel or set a timeout for the HTTP request
}
```

✅ GOOD:
```go
func fetchData(ctx context.Context, url string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    resp, err := http.DefaultClient.Do(req) // Respects context cancellation/timeout
    if err != nil {
        return nil, fmt.Errorf("http request failed: %w", err)
    }
    defer resp.Body.Close()
    return io.ReadAll(resp.Body)
}
```

### Coordinate Goroutines with `errgroup.Group`
For managing multiple goroutines that need to complete or be cancelled together, use `golang.org/x/sync/errgroup`. This simplifies error propagation and context cancellation across concurrent tasks.

❌ BAD:
```go
var wg sync.WaitGroup
var mu sync.Mutex
var errors []error // Manual error collection

for _, task := range tasks {
    wg.Add(1)
    go func(t Task) {
        defer wg.Done()
        if err := t.Run(); err != nil {
            mu.Lock()
            errors = append(errors, err)
            mu.Unlock()
        }
    }(task)
}
wg.Wait()
// Manual error checking and potential resource leaks on first error
```

✅ GOOD:
```go
group, ctx := errgroup.WithContext(context.Background()) // Context for cancellation
// var results []Result // Use a channel or mutex-protected slice if results are needed

for _, task := range tasks {
    task := task // Capture loop variable for closure
    group.Go(func() error {
        select {
        case <-ctx.Done(): // Respect cancellation from other goroutines
            return ctx.Err()
        default:
            if err := task.Run(ctx); err != nil { // Pass context to task
                return fmt.Errorf("task %s failed: %w", task.ID, err)
            }
            return nil
        }
    })
}

if err := group.Wait(); err != nil { // Waits for all, returns first non-nil error
    log.Printf("One or more tasks failed: %v", err)
}
```

## 5. API Design

**Design clear, intuitive, and idiomatic APIs.** Follow Go's conventions for function signatures and return values.

### Idiomatic Function Signatures
Prefer `(value, error)` return patterns. Avoid naked returns and ensure clarity in parameter order (e.g., `context.Context` first, then inputs, then options).

❌ BAD:
```go
func GetUser(id string) (user User, err error) { // Naked return, less clear
    // ...
    return
}
```

✅ GOOD:
```go
func GetUser(ctx context.Context, id string) (User, error) { // Clear return, context first
    // ...
    return user, nil
}
```

### Accept Interfaces, Return Structs
This principle promotes flexibility for callers (they can pass any type satisfying the interface) while maintaining concrete implementation details internally.

❌ BAD:
```go
func Process(reader *bytes.Reader) error { // Too specific, limits caller flexibility
    // ...
}
```

✅ GOOD:
```go
func Process(reader io.Reader) error { // Accepts any io.Reader, highly flexible
    // ...
}
```

## 6. Testing Approaches

**Write comprehensive, fast, and maintainable tests.** Prioritize table-driven tests for clarity and coverage.

### Table-Driven Tests
Use table-driven tests for functions with multiple inputs and expected outputs. This reduces boilerplate, improves readability, and makes it easy to add new test cases.

❌ BAD:
```go
func TestAdd(t *testing.T) {
    if Add(1, 2) != 3 {
        t.Errorf("Add(1, 2) = %d; want 3", Add(1, 2))
    }
    if Add(-1, 1) != 0 {
        t.Errorf("Add(-1, 1) = %d; want 0", Add(-1, 1))
    }
}
```

✅ GOOD:
```go
func TestAdd(t *testing.T) {
    tests := []struct {
        name string
        a, b int
        want int
    }{
        {"positive numbers", 1, 2, 3},
        {"negative and positive", -1, 1, 0},
        {"zero", 0, 0, 0},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) { // Use t.Run for isolated subtests
            if got := Add(tt.a, tt.b); got != tt.want {
                t.Errorf("Add(%d, %d) = %d; want %d", tt.a, tt.b, got, tt.want)
            }
        })
    }
}
```

## 7. Security Best Practices

**Integrate security from the start.** Always validate inputs and handle sensitive data carefully.

### Input Validation
Never trust user input. Validate all inputs at the API boundary to prevent injection attacks, buffer overflows, and other vulnerabilities.

❌ BAD:
```go
func createUser(username string) error {
    // Directly uses username from request without validation
    db.CreateUser(username)
}
```

✅ GOOD:
```go
func createUser(username string) error {
    if !isValidUsername(username) { // Validate length, allowed characters, format
        return errors.New("invalid username format")
    }
    db.CreateUser(username)
}
```

### Secure Dependency Management
Regularly audit and update dependencies to mitigate known vulnerabilities. Use `go mod tidy` to clean up unused modules and `govulncheck` to identify issues.

❌ BAD:
```go
// Relying on old, potentially vulnerable dependencies without checks
```

✅ GOOD:
```go
// Keep go.mod clean and dependencies updated.
// Use `go get -u ./...` and `go mod tidy`.
// Regularly run `govulncheck ./...` to identify known vulnerabilities.
```