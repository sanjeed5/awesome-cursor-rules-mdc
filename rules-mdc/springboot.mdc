---
description: Enforces modern Spring Boot best practices for Java applications, covering code structure, dependency injection, API design, error handling, and testing to ensure maintainable, performant, and secure microservices.
globs: **/*.java
---
# springboot Best Practices

This guide outlines the definitive best practices for developing Spring Boot 3.x applications with Java 17+. Adhere to these rules to build robust, maintainable, and performant services.

## 1. Code Organization and Structure

Organize your codebase by feature or bounded context, not by technical layer. This improves navigability, cohesion, and testability.

❌ BAD: Technical Layering
```
com.myapp.project.controller.UserController
com.myapp.project.service.UserService
com.myapp.project.repository.UserRepository
```

✅ GOOD: Feature-based (Bounded Context)
```java
// Root application class in base package
package com.myapp.project;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ProjectApplication {
    public static void main(String[] args) {
        SpringApplication.run(ProjectApplication.class, args);
    }
}

// User feature package
package com.myapp.project.user;

// User feature sub-packages
package com.myapp.project.user.api; // REST controllers, DTOs
package com.myapp.project.user.domain; // Business logic, entities
package com.myapp.project.user.infrastructure; // Repositories, external integrations
```

## 2. Dependency Management

Always inherit from `spring-boot-starter-parent` or use `spring-boot-dependencies` BOM for consistent dependency versions.

✅ GOOD: Maven `pom.xml`
```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.11</version> <!-- Use latest stable version -->
    <relativePath/>
</parent>
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!-- No version needed for Spring Boot managed dependencies -->
</dependencies>
```

## 3. Dependency Injection

Prefer constructor injection for all dependencies. This ensures immutability, simplifies testing, and makes dependencies explicit. Use Lombok's `@RequiredArgsConstructor` for conciseness.

❌ BAD: Field Injection
```java
@Service
public class UserService {
    @Autowired // Avoid field injection
    private UserRepository userRepository;
}
```

✅ GOOD: Constructor Injection with Lombok
```java
package com.myapp.project.user.domain;

import com.myapp.project.user.infrastructure.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor // Automatically generates constructor for final fields
public class UserService {
    private final UserRepository userRepository; // Make dependencies final

    public User findUserById(Long id) {
        return userRepository.findById(id).orElseThrow(() -> new UserNotFoundException(id));
    }
}
```

## 4. API Design and Controllers

Controllers must be stateless and focused solely on routing HTTP requests and responses. Delegate all business logic to service layers. Use DTOs for request/response bodies.

❌ BAD: Business Logic in Controller
```java
@RestController
@RequestMapping("/api/users")
public class UserController {
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        // Business logic directly in controller - BAD
        if (user.getEmail() == null || !user.getEmail().contains("@")) {
            throw new IllegalArgumentException("Invalid email");
        }
        // ... more logic
        return ResponseEntity.ok(userRepository.save(user));
    }
}
```

✅ GOOD: Lean Controller, Delegate to Service
```java
package com.myapp.project.user.api;

import com.myapp.project.user.domain.UserService;
import com.myapp.project.user.api.dto.UserCreateRequest;
import com.myapp.project.user.api.dto.UserResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {
    private final UserService userService;

    @PostMapping
    public ResponseEntity<UserResponse> createUser(@RequestBody UserCreateRequest request) {
        UserResponse response = userService.createUser(request);
        return new ResponseEntity<>(response, HttpStatus.CREATED);
    }

    @GetMapping("/{id}")
    public ResponseEntity<UserResponse> getUserById(@PathVariable Long id) {
        UserResponse response = userService.findUserById(id);
        return ResponseEntity.ok(response);
    }
}
```

## 5. Logging

Use SLF4J with Logback (Spring Boot's default) for all logging. Avoid `System.out.print()`. Use Lombok's `@Slf4j` for convenience. Avoid string concatenation in log messages; use parameterized logging.

❌ BAD: `System.out.print()` and String Concatenation
```java
public void process(String data) {
    System.out.println("Processing data: " + data); // BAD
    log.info("Processing data: " + data); // Still BAD for performance
}
```

✅ GOOD: Parameterized SLF4J Logging
```java
package com.myapp.project.user.domain;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@Slf4j // Provides 'log' field
public class UserService {
    public void processData(String data) {
        log.info("Processing data: {}", data); // Use placeholders for performance
        try {
            // ... business logic
        } catch (Exception e) {
            log.error("Failed to process data: {}", data, e); // Include exception
        }
    }
}
```

## 6. Error Handling

Implement global exception handling using `@RestControllerAdvice` to provide consistent and meaningful error responses.

✅ GOOD: Global Exception Handler
```java
package com.myapp.project.common.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

@ControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<Object> handleUserNotFoundException(UserNotFoundException ex, WebRequest request) {
        ErrorResponse error = new ErrorResponse(HttpStatus.NOT_FOUND, ex.getMessage());
        return new ResponseEntity<>(error, HttpStatus.NOT_FOUND);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Object> handleIllegalArgumentException(IllegalArgumentException ex, WebRequest request) {
        ErrorResponse error = new ErrorResponse(HttpStatus.BAD_REQUEST, ex.getMessage());
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    // Define custom error response structure
    record ErrorResponse(HttpStatus status, String message) {}
}

// Custom exception example
package com.myapp.project.user.domain;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

@ResponseStatus(HttpStatus.NOT_FOUND) // Can also be handled by @ControllerAdvice
public class UserNotFoundException extends RuntimeException {
    public UserNotFoundException(Long id) {
        super("Could not find user " + id);
    }
}
```

## 7. Configuration

Store all external configuration in `application.yml` or environment variables. Avoid hardcoding values. Use `@ConfigurationProperties` for type-safe configuration.

✅ GOOD: `application.yml`
```yaml
app:
  service:
    baseUrl: https://api.example.com
    timeoutMs: 5000
```

✅ GOOD: Type-Safe Configuration Class
```java
package com.myapp.project.common.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConfigurationProperties(prefix = "app.service")
@Data // Lombok for getters/setters
public class AppServiceProperties {
    private String baseUrl;
    private int timeoutMs;
}

// Usage
@Service
@RequiredArgsConstructor
public class ExternalServiceClient {
    private final AppServiceProperties properties;

    public void callExternalService() {
        // Use properties.getBaseUrl(), properties.getTimeoutMs()
    }
}
```

## 8. Testing

Write comprehensive unit and integration tests. Leverage Spring Boot's testing utilities. Constructor injection greatly aids unit testing.

✅ GOOD: Unit Test (Service Layer)
```java
package com.myapp.project.user.domain;

import com.myapp.project.user.infrastructure.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;

class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService; // Injects mocks into UserService

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void findUserById_userExists_returnsUser() {
        User mockUser = new User(1L, "test@example.com");
        when(userRepository.findById(1L)).thenReturn(Optional.of(mockUser));

        User foundUser = userService.findUserById(1L);
        // Assertions...
    }

    @Test
    void findUserById_userNotFound_throwsException() {
        when(userRepository.findById(2L)).thenReturn(Optional.empty());

        assertThrows(UserNotFoundException.class, () -> userService.findUserById(2L));
    }
}
```

✅ GOOD: Integration Test (Controller Layer)
```java
package com.myapp.project.user.api;

import com.myapp.project.user.domain.UserService;
import com.myapp.project.user.api.dto.UserCreateRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(UserController.class) // Focuses on Web layer, mocks other beans
class UserControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean // Mocks the UserService bean
    private UserService userService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void createUser_validRequest_returnsCreated() throws Exception {
        UserCreateRequest request = new UserCreateRequest("newuser@example.com", "password");
        when(userService.createUser(any(UserCreateRequest.class)))
            .thenReturn(new UserResponse(1L, "newuser@example.com"));

        mockMvc.perform(post("/api/users")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated());
    }
}
```