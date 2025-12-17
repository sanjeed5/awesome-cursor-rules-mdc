---
description: This rule file provides definitive guidelines for using Go's `http.ServeMux` effectively, leveraging Go 1.22+ features for robust, maintainable, and secure API development.
globs: **/*.go
---
# servemux Best Practices

Go's `http.ServeMux`, especially with the enhancements in Go 1.22+, is the definitive choice for building performant and maintainable HTTP services. This guide outlines the best practices for its use.

## 1. Code Organization and Structure

Organize your application for clarity, testability, and scalability.

### ✅ GOOD: Centralized Mux Assembly, Dedicated Handlers

Create a single `http.ServeMux` instance in your application's entry point (`cmd/server/main.go`) and register handlers from a dedicated `handlers` package. Inject business logic into handlers via interfaces.

```go
// cmd/server/main.go
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"yourproject/internal/handlers"
	"yourproject/internal/service" // Business logic interface
)

func main() {
	logger := log.New(os.Stdout, "API: ", log.Ldate|log.Ltime|log.Lshortfile)

	// Initialize business logic (e.g., database connection)
	userService := service.NewUserService(logger) // Assume concrete implementation

	// Create and configure the mux
	mux := http.NewServeMux()
	handlers.RegisterRoutes(mux, logger, userService) // Pass dependencies

	server := &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
		ErrorLog:     logger,
	}

	// Start server in a goroutine
	go func() {
		logger.Printf("Server starting on %s", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("Could not listen on %s: %v\n", server.Addr, err)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	logger.Println("Server shutting down...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Fatalf("Server forced to shutdown: %v", err)
	}
	logger.Println("Server exited gracefully")
}

// internal/handlers/user.go
package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"yourproject/internal/service" // Business logic interface
)

// UserService defines the interface for user-related business logic.
type UserService interface {
	GetUser(id int) (*service.User, error)
	CreateUser(user *service.User) error
	// ... other user methods
}

// UserHandlers holds dependencies for user-related HTTP handlers.
type UserHandlers struct {
	log *log.Logger
	svc UserService
}

// RegisterRoutes registers all user-related HTTP routes with the given mux.
func RegisterRoutes(mux *http.ServeMux, logger *log.Logger, userService UserService) {
	uh := &UserHandlers{
		log: logger,
		svc: userService,
	}

	mux.HandleFunc("GET /users/{id}", uh.GetUser)
	mux.HandleFunc("POST /users", uh.CreateUser)
	// ... other route registrations
}

// GetUser handles GET requests for /users/{id}
func (uh *UserHandlers) GetUser(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		uh.respondWithError(w, http.StatusBadRequest, "Invalid user ID")
		return
	}

	user, err := uh.svc.GetUser(id)
	if err != nil {
		uh.respondWithError(w, http.StatusNotFound, "User not found")
		return
	}

	uh.respondWithJSON(w, http.StatusOK, user)
}

// CreateUser handles POST requests for /users
func (uh *UserHandlers) CreateUser(w http.ResponseWriter, r *http.Request) {
	var user service.User // Assuming service.User is the DTO
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		uh.respondWithError(w, http.StatusBadRequest, "Invalid request payload")
		return
	}

	if err := uh.svc.CreateUser(&user); err != nil {
		uh.respondWithError(w, http.StatusInternalServerError, "Failed to create user")
		return
	}

	uh.respondWithJSON(w, http.StatusCreated, user)
}

// respondWithError sends a JSON error response.
func (uh *UserHandlers) respondWithError(w http.ResponseWriter, code int, message string) {
	uh.respondWithJSON(w, code, map[string]string{"error": message})
}

// respondWithJSON sends a JSON response.
func (uh *UserHandlers) respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, err := json.Marshal(payload)
	if err != nil {
		uh.log.Printf("Error marshaling JSON response: %v", err)
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(response)
}
```

### ❌ BAD: Global Mux, Logic in `main`, Repetitive Naming

Avoid using `http.DefaultServeMux` and polluting `main.go` with handler logic. Do not repeat package or receiver names in function signatures.

```go
// main.go (BAD example)
package main

import (
	"fmt"
	"log"
	"net/http"
	"strconv"
)

// Bad naming: GetUserHandler repeats "User" and "Handler"
func GetUserHandler(w http.ResponseWriter, r *http.Request) {
	idStr := r.PathValue("id")
	id, err := strconv.Atoi(idStr)
	if err != nil {
		http.Error(w, "Invalid ID", http.StatusBadRequest)
		return
	}
	// Business logic directly here, hard to test
	fmt.Fprintf(w, "Getting user %d\n", id)
}

func main() {
	// Bad: Using http.HandleFunc directly, relies on global DefaultServeMux
	http.HandleFunc("GET /users/{id}", GetUserHandler)
	log.Fatal(http.ListenAndServe(":8080", nil)) // nil uses DefaultServeMux
}
```

## 2. API Design and Routing (Go 1.22+ Features)

Leverage Go 1.22's enhanced `http.ServeMux` for clean, expressive routing.

### ✅ GOOD: Method Matching, Path Values, and Exact Matches

Use HTTP method prefixes, `{name}` wildcards for path values, and `{$}` for strict path matching.

```go
// In handlers.RegisterRoutes or similar setup
mux.HandleFunc("GET /users/{id}", uh.GetUser)       // Matches GET /users/123, extracts "123" as "id"
mux.HandleFunc("POST /users", uh.CreateUser)        // Matches POST /users
mux.HandleFunc("DELETE /users/{id}", uh.DeleteUser) // Matches DELETE /users/123
mux.HandleFunc("GET /healthz/{$}", uh.HealthCheck)  // Matches ONLY GET /healthz, not /healthz/foo
mux.HandleFunc("/files/", uh.FileServer)            // Matches /files/ and /files/foo/bar (subtree)
```

### ❌ BAD: Manual Method Checking, Suboptimal Path Matching

Avoid checking `r.Method` inside handlers or using generic path patterns when specific methods are required.

```go
// In handlers.RegisterRoutes or similar setup (BAD example)
mux.HandleFunc("/users/{id}", uh.UserHandlerGeneric) // Handles all methods for /users/{id}

// func (uh *UserHandlers) UserHandlerGeneric(w http.ResponseWriter, r *http.Request) {
// 	switch r.Method { // BAD: Manual method dispatch
// 	case http.MethodGet:
// 		uh.GetUser(w, r)
// 	case http.MethodDelete:
// 		uh.DeleteUser(w, r)
// 	default:
// 		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
// 	}
// }

mux.HandleFunc("/healthz/", uh.HealthCheck) // BAD: Matches /healthz/foo, not just /healthz
```

## 3. Error Handling and Response Encoding

Always return structured JSON error responses and use `json.NewEncoder` for output.

### ✅ GOOD: JSON Error Responses, `json.NewEncoder`

Define a consistent error response structure. Use `json.NewEncoder` for efficient and correct JSON output.

```go
// In handlers/user.go (example helper functions)

// respondWithError sends a JSON error response.
func (uh *UserHandlers) respondWithError(w http.ResponseWriter, code int, message string) {
	uh.respondWithJSON(w, code, map[string]string{"error": message})
}

// respondWithJSON sends a JSON response.
func (uh *UserHandlers) respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(payload); err != nil { // Use NewEncoder
		uh.log.Printf("Error encoding JSON response: %v", err)
		// Fallback to plain text error if JSON encoding fails
		http.Error(w, "Internal Server Error", http.StatusInternalServerError)
	}
}
```

### ❌ BAD: Plain Text Errors, Manual JSON Marshaling

Avoid `http.Error` for API responses and manual `json.Marshal` followed by `w.Write`.

```go
// func (uh *UserHandlers) GetUser(w http.ResponseWriter, r *http.Request) {
// 	// ...
// 	if err != nil {
// 		http.Error(w, "User not found", http.StatusNotFound) // BAD: Plain text error
// 		return
// 	}
//
// 	response, _ := json.Marshal(user) // BAD: Manual Marshal, then Write
// 	w.Header().Set("Content-Type", "application/json")
// 	w.WriteHeader(http.StatusOK)
// 	w.Write(response)
// }
```

## 4. Logging

Implement clear, request-level logging using the standard `log` package or a lightweight wrapper.

### ✅ GOOD: Request-Level Logging

Log incoming requests and critical events within handlers.

```go
// In handlers/user.go
func (uh *UserHandlers) GetUser(w http.ResponseWriter, r *http.Request) {
	uh.log.Printf("INFO: Handling GET /users/%s from %s", r.PathValue("id"), r.RemoteAddr)
	// ... handler logic
}
```

## 5. Testing Approaches

Write comprehensive unit tests for each handler using `httptest.NewRecorder`.

### ✅ GOOD: `httptest.NewRecorder` for Handler Unit Tests

Test handlers in isolation, validating status codes, headers, and response bodies.

```go
// internal/handlers/user_test.go
package handlers_test

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"yourproject/internal/handlers"
	"yourproject/internal/service" // Business logic interface
)

// MockUserService implements handlers.UserService for testing purposes.
type MockUserService struct {
	GetUserFn    func(id int) (*service.User, error)
	CreateUserFn func(user *service.User) error
}

func (m *MockUserService) GetUser(id int) (*service.User, error) {
	return m.GetUserFn(id)
}
func (m *MockUserService) CreateUser(user *service.User) error {
	return m.CreateUserFn(user)
}

func TestGetUser(t *testing.T) {
	tests := []struct {
		name           string
		userID         string
		mockGetUser    func(id int) (*service.User, error)
		expectedStatus int
		expectedBody   string
	}{
		{
			name:   "Valid User ID",
			userID: "1",
			mockGetUser: func(id int) (*service.User, error) {
				return &service.User{ID: 1, Name: "Test User"}, nil
			},
			expectedStatus: http.StatusOK,
			expectedBody:   `{"ID":1,"Name":"Test User"}`,
		},
		{
			name:           "User Not Found",
			userID: "999",
			mockGetUser: func(id int) (*service.User, error) {
				return nil, service.ErrNotFound // Assume service.ErrNotFound
			},
			expectedStatus: http.StatusNotFound,
			expectedBody:   `{"error":"User not found"}`,
		},
		{
			name:           "Invalid User ID",
			userID: "abc",
			mockGetUser:    nil, // Should not be called
			expectedStatus: http.StatusBadRequest,
			expectedBody:   `{"error":"Invalid user ID"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockSvc := &MockUserService{GetUserFn: tt.mockGetUser}
			mux := http.NewServeMux()
			handlers.RegisterRoutes(mux, log.Default(), mockSvc) // Use default logger for tests

			req := httptest.NewRequest(http.MethodGet, "/users/"+tt.userID, nil)
			rr := httptest.NewRecorder()

			mux.ServeHTTP(rr, req) // Call the mux directly

			if rr.Code != tt.expectedStatus {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, rr.Code)
			}

			// Compare JSON bodies
			var actual map[string]interface{}
			var expected map[string]interface{}
			json.Unmarshal(rr.Body.Bytes(), &actual)
			json.Unmarshal([]byte(tt.expectedBody), &expected)

			if !bytes.Equal(rr.Body.Bytes(), []byte(tt.expectedBody)) && !jsonEqual(actual, expected) {
				t.Errorf("Expected body %s, got %s", tt.expectedBody, rr.Body.String())
			}
		})
	}
}

// jsonEqual compares two JSON objects for equality.
func jsonEqual(a, b interface{}) bool {
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	return bytes.Equal(aj, bj)
}
```

## 6. Security Best Practices

Prioritize security by avoiding global state and validating all inputs.

### ✅ GOOD: Use a Local `http.ServeMux`

Always instantiate your own `http.ServeMux` instance. This prevents third-party packages from inadvertently or maliciously registering routes on your server.

```go
// main.go
mux := http.NewServeMux() // ✅ GOOD: Local, controlled mux
// ... register handlers
http.ListenAndServe(":8080", mux)
```

### ❌ BAD: Relying on `http.DefaultServeMux`

Never pass `nil` to `http.ListenAndServe()` or use `http.Handle`/`http.HandleFunc` directly, as this exposes the global `http.DefaultServeMux`.

```go
// main.go (BAD example)
http.HandleFunc("/", homeHandler) // BAD: Registers on global mux
http.ListenAndServe(":8080", nil) // BAD: Uses global mux
```

## 7. Common Pitfalls and Gotchas

Be aware of `servemux`'s specific behaviors to avoid unexpected issues.

### ✅ GOOD: Understand Pattern Precedence and Conflicts

Go 1.22+ `ServeMux` panics on registration if patterns conflict. Design your routes to be unambiguous. Use `{$}` for exact matches to prevent subtree matching.

```go
// Correctly defined patterns
mux.HandleFunc("GET /tasks/{id}/status", handlerTaskStatus) // Specific
mux.HandleFunc("GET /tasks/{id}", handlerTask)             // More general, but no conflict
mux.HandleFunc("GET /admin/{$}", handlerAdminRoot)         // Exact match for /admin
```

### ❌ BAD: Ambiguous Patterns, Forgetting `{$}`

```go
// mux.HandleFunc("/tasks/{id}/status/", handlerTaskStatus) // BAD: Trailing slash makes it a subtree
// mux.HandleFunc("/tasks/0/{action}/", handlerTaskAction)  // BAD: Conflicts with above for /tasks/0/status/
//
// mux.HandleFunc("/admin/", handlerAdminRoot) // BAD: Matches /admin/foo, not just /admin
```