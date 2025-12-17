---
description: Definitive guidelines for writing high-performance, secure, and maintainable Fiber applications in Go. Focuses on context immutability, robust error handling, and modular design.
globs: **/*.go
---
# Fiber Best Practices

Fiber is a high-performance, Express-inspired web framework for Go, built on `fasthttp`. These rules ensure we leverage Fiber's speed and maintainability while avoiding common pitfalls.

## 1. Core Principle: `fiber.Ctx` Immutability

**NEVER** retain references to `fiber.Ctx` or any data extracted from it (like `c.Params`, `c.Query`, `c.Body`) after the handler returns. `fiber.Ctx` values are reused across requests, making their contents ephemeral. Always copy data if you need to store it.

❌ BAD: Retaining ephemeral data

```go
package handlers

import (
	"github.com/gofiber/fiber/v2"
	"log"
)

// This struct will hold a reference to the ephemeral string
type UserRequest struct {
	UserID string
}

func GetUserBad(c *fiber.Ctx) error {
	// DANGER: c.Params("id") returns an ephemeral string.
	// Storing it directly will lead to data corruption in future requests.
	req := &UserRequest{
		UserID: c.Params("id"),
	}

	// Simulating async processing or storage
	go func() {
		// By the time this goroutine runs, c.Params("id") might have changed
		log.Printf("Processing user ID: %s", req.UserID) // Will likely be wrong
	}()

	return c.SendString("Request received (bad example)")
}
```

✅ GOOD: Copying ephemeral data

```go
package handlers

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/utils" // Use Fiber's utility for efficient copying
	"log"
)

// This struct holds a safe copy of the string
type UserRequest struct {
	UserID string
}

func GetUserGood(c *fiber.Ctx) error {
	// ALWAYS copy data from c.Params, c.Query, c.Body, etc.
	userID := utils.CopyString(c.Params("id"))
	req := &UserRequest{
		UserID: userID,
	}

	// Simulating async processing or storage
	go func() {
		// This is safe; userID is a distinct copy
		log.Printf("Processing user ID: %s", req.UserID) // Will always be correct
	}()

	return c.SendString("Request received (good example)")
}
```

## 2. Code Organization and Structure

Organize your application into logical layers: `main`, `config`, `router`, `middleware`, `handlers`, `services`, `repository`, `models`, `utils`.

```
.
├── cmd/api/main.go          # Application entry point
├── config/config.go         # Application configuration loading
├── internal/
│   ├── handlers/            # HTTP request handlers
│   │   └── user_handler.go
│   ├── middleware/          # Global and route-specific middleware
│   │   └── security.go
│   ├── models/              # Data structures (structs for DB, JSON, etc.)
│   │   └── user.go
│   ├── repository/          # Database interaction logic
│   │   └── user_repo.go
│   ├── router/              # Centralized route definitions
│   │   └── router.go
│   └── services/            # Business logic layer
│       └── user_service.go
└── pkg/
    └── utils/               # Reusable utility functions
        └── response.go
```

## 3. Security Best Practices

Always apply essential security middleware globally.

```go
package middleware

import (
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/csrf"
	"github.com/gofiber/fiber/v2/middleware/helmet"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gofiber/fiber/v2/utils"
)

// FiberMiddleware provides essential security and utility middleware.
func FiberMiddleware(app *fiber.App) {
	app.Use(
		// Recover from panics to prevent server crashes
		recover.New(),
		// Add security headers
		helmet.New(),
		// Enable CORS for frontend integration
		cors.New(),
		// Protect against CSRF attacks
		csrf.New(csrf.Config{
			KeyLookup:      "header:X-Csrf-Token",
			CookieName:     "__Host-csrf_",
			CookieSameSite: "Strict",
			Expiration:     3 * time.Hour,
			KeyGenerator:   utils.UUID,
		}),
		// Rate limit requests to prevent abuse
		limiter.New(limiter.Config{
			Max:        20,
			Expiration: 30 * time.Second,
			// KeyGenerator: func(c *fiber.Ctx) string { return c.IP() }, // Customize if needed
		}),
		// Compress responses
		compress.New(),
		// Request logging
		logger.New(),
	)
}
```

## 4. Error Handling

Implement a centralized error handler and use custom error types for clarity.

❌ BAD: Inconsistent error responses

```go
// handler.go
func GetUser(c *fiber.Ctx) error {
	id := c.Params("id")
	if id == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "ID is required"})
	}
	// ... database error
	return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Database error"})
}
```

✅ GOOD: Centralized error handling with custom errors

```go
package utils

import "github.com/gofiber/fiber/v2"

// AppError represents a custom application error
type AppError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Err     error  `json:"-"` // Original error for logging, not exposed to client
}

func (e *AppError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}
	return e.Message
}

// NewAppError creates a new AppError
func NewAppError(code int, message string, err error) *AppError {
	return &AppError{
		Code:    code,
		Message: message,
		Err:     err,
	}
}

// GlobalErrorHandler handles all errors in the application
func GlobalErrorHandler(c *fiber.Ctx, err error) error {
	// Default error response
	code := fiber.StatusInternalServerError
	message := "Internal Server Error"

	if e, ok := err.(*AppError); ok {
		code = e.Code
		message = e.Message
		// Log the original error for debugging
		if e.Err != nil {
			c.App().Logger().Errorf("AppError: %s, Original: %v", e.Message, e.Err)
		}
	} else if e, ok := err.(*fiber.Error); ok {
		code = e.Code
		message = e.Message
	} else {
		// Log unexpected errors
		c.App().Logger().Errorf("Unhandled error: %v", err)
	}

	return c.Status(code).JSON(fiber.Map{
		"code":    code,
		"message": message,
	})
}
```

```go
// In your main.go or router setup:
app := fiber.New(fiber.Config{
	ErrorHandler: utils.GlobalErrorHandler,
})

// In your handler:
func GetUser(c *fiber.Ctx) error {
	id := c.Params("id")
	if id == "" {
		return utils.NewAppError(fiber.StatusBadRequest, "User ID is required", nil)
	}
	// Simulate a service call that might return an error
	user, err := userService.GetUserByID(id)
	if err != nil {
		// Wrap service errors into AppError for consistent client response
		return utils.NewAppError(fiber.StatusInternalServerError, "Failed to retrieve user", err)
	}
	return c.JSON(user)
}
```

## 5. API Design: RESTful & Consistent Responses

Design your APIs following REST principles and ensure consistent JSON response structures.

❌ BAD: Inconsistent response formats

```go
// GET /users/:id
// Success: {"id": "123", "name": "John Doe"}
// Error: "User not found" (plain text)

// POST /users
// Success: "User created" (plain text)
// Error: {"message": "Invalid input"} (JSON)
```

✅ GOOD: Standardized JSON responses

```go
package utils

// SuccessResponse for successful API calls
type SuccessResponse struct {
	Status  int         `json:"status"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// SendSuccess sends a standardized success response
func SendSuccess(c *fiber.Ctx, status int, message string, data interface{}) error {
	return c.Status(status).JSON(SuccessResponse{
		Status:  status,
		Message: message,
		Data:    data,
	})
}
```

```go
// In your handler:
func GetUser(c *fiber.Ctx) error {
	id := c.Params("id")
	user, err := userService.GetUserByID(id)
	if err != nil {
		// Error handling via GlobalErrorHandler
		return utils.NewAppError(fiber.StatusNotFound, "User not found", err)
	}
	return utils.SendSuccess(c, fiber.StatusOK, "User retrieved successfully", user)
}

func CreateUser(c *fiber.Ctx) error {
	user := new(models.User)
	if err := c.BodyParser(user); err != nil {
		return utils.NewAppError(fiber.StatusBadRequest, "Invalid request body", err)
	}
	// ... validation and service call
	createdUser, err := userService.CreateUser(user)
	if err != nil {
		return utils.NewAppError(fiber.StatusInternalServerError, "Failed to create user", err)
	}
	return utils.SendSuccess(c, fiber.StatusCreated, "User created successfully", createdUser)
}
```

## 6. Testing Approaches

Use `httptest` and Fiber's test utilities for robust, table-driven unit tests. Inject dependencies to enable easy mocking.

```go
package handlers_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v2"
	"github.com/stretchr/testify/assert"

	"your_module/internal/handlers"
	"your_module/internal/models"
	"your_module/internal/services" // Mock this
	"your_module/pkg/utils"
)

// MockUserService implements the UserService interface for testing
type MockUserService struct {
	GetUserByIDFunc func(id string) (*models.User, error)
	CreateUserFunc  func(user *models.User) (*models.User, error)
}

func (m *MockUserService) GetUserByID(id string) (*models.User, error) {
	return m.GetUserByIDFunc(id)
}
func (m *MockUserService) CreateUser(user *models.User) (*models.User, error) {
	return m.CreateUserFunc(user)
}

func TestGetUser(t *testing.T) {
	app := fiber.New(fiber.Config{
		ErrorHandler: utils.GlobalErrorHandler, // Use global error handler for consistency
	})

	// Setup mock service
	mockService := &MockUserService{}
	userHandler := handlers.NewUserHandler(mockService) // Inject mock service

	app.Get("/users/:id", userHandler.GetUser)

	tests := []struct {
		name           string
		userID         string
		mockReturnUser *models.User
		mockReturnErr  error
		expectedStatus int
		expectedBody   string
	}{
		{
			name:           "Success",
			userID:         "1",
			mockReturnUser: &models.User{ID: "1", Name: "Test User"},
			mockReturnErr:  nil,
			expectedStatus: fiber.StatusOK,
			expectedBody:   `{"status":200,"message":"User retrieved successfully","data":{"id":"1","name":"Test User"}}`,
		},
		{
			name:           "Not Found",
			userID:         "2",
			mockReturnUser: nil,
			mockReturnErr:  errors.New("user not found"),
			expectedStatus: fiber.StatusNotFound,
			expectedBody:   `{"code":404,"message":"User not found"}`,
		},
		{
			name:           "Invalid ID",
			userID:         "", // Handled by handler logic
			mockReturnUser: nil,
			mockReturnErr:  nil, // This won't be called if ID is empty
			expectedStatus: fiber.StatusBadRequest,
			expectedBody:   `{"code":400,"message":"User ID is required"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockService.GetUserByIDFunc = func(id string) (*models.User, error) {
				assert.Equal(t, tt.userID, id) // Ensure correct ID is passed
				return tt.mockReturnUser, tt.mockReturnErr
			}

			req := httptest.NewRequest(fiber.MethodGet, "/users/"+tt.userID, nil)
			resp, _ := app.Test(req, -1) // -1 disables request latency

			assert.Equal(t, tt.expectedStatus, resp.StatusCode)

			body, _ := io.ReadAll(resp.Body)
			assert.JSONEq(t, tt.expectedBody, string(body))
		})
	}
}
```