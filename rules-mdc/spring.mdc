---
description: Definitive guidelines for building robust, maintainable, and performant Spring Boot applications using modern best practices.
globs: **/*.java
---
# Spring Best Practices

This guide outlines essential best practices for developing with Spring Boot, ensuring your applications are well-structured, performant, secure, and easily testable.

## 1. Code Organization and Structure

Maintain a clean, logical package structure.

### 1.1. Base Package and `@SpringBootApplication`

Place your main application class at the root of a well-named base package. All components should reside within this base package or its sub-packages.

❌ BAD:
```java
// com.example.Application.java
package com.example; // Too generic, or in default package
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

✅ GOOD:
```java
// com.company.project.MyApplication.java
package com.company.project; // Specific base package
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MyApplication { // Clear application entry point
    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }
}
```

### 1.2. Configuration Classes

Group related bean definitions into dedicated `@Configuration` classes.

❌ BAD:
```java
// MyApplication.java (monolithic config)
@SpringBootApplication
public class MyApplication {
    // ...
    @Bean
    public DataSource dataSource() { /* ... */ }
    @Bean
    public RestTemplate restTemplate() { /* ... */ }
}
```

✅ GOOD:
```java
// com.company.project.config.DataSourceConfig.java
@Configuration
public class DataSourceConfig {
    @Bean
    public DataSource dataSource() { /* ... */ }
}

// com.company.project.config.WebClientConfig.java
@Configuration
public class WebClientConfig {
    @Bean
    public WebClient webClient(WebClient.Builder builder) {
        return builder.baseUrl("http://api.example.com").build();
    }
}
```

## 2. Common Patterns and Anti-patterns

### 2.1. Constructor Injection

Always prefer constructor injection for mandatory dependencies. It ensures immutability and testability.

❌ BAD:
```java
@Service
public class UserService {
    @Autowired // Field injection is discouraged
    private UserRepository userRepository;

    public User findById(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
```

✅ GOOD:
```java
@Service
public class UserService {
    private final UserRepository userRepository; // Immutable

    @Autowired // Optional for single constructor in Spring 4.3+
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public User findById(Long id) {
        return userRepository.findById(id).orElse(null);
    }
}
```

### 2.2. Immutable Data Models

Use immutable classes for DTOs and entities where possible, especially for value objects.

❌ BAD:
```java
public class UserDto { // Mutable
    private Long id;
    private String name;
    // Getters and Setters
}
```

✅ GOOD:
```java
public record UserDto(Long id, String name) {} // Immutable record (Java 16+)
// Or with Lombok:
@Value // All-args constructor, getters, equals, hashCode, toString
public class UserDto {
    Long id;
    String name;
}
```

## 3. Performance Considerations

### 3.1. N+1 Query Problem

Avoid N+1 queries by eagerly fetching related data when necessary.

❌ BAD:
```java
@Service
public class OrderService {
    @Autowired private OrderRepository orderRepository;
    public List<Order> getOrdersWithItems() {
        List<Order> orders = orderRepository.findAll();
        orders.forEach(order -> order.getItems().size()); // N+1 queries for items
        return orders;
    }
}
```

✅ GOOD:
```java
@Repository
public interface OrderRepository extends JpaRepository<Order, Long> {
    @Query("SELECT o FROM Order o JOIN FETCH o.items") // Eagerly fetch items
    List<Order> findAllWithItems();
}

@Service
public class OrderService {
    @Autowired private OrderRepository orderRepository;
    public List<Order> getOrdersWithItems() {
        return orderRepository.findAllWithItems(); // Single query
    }
}
```

## 4. Common Pitfalls and Gotchas

### 4.1. Disabling Auto-configuration

Only disable auto-configuration when strictly necessary to avoid unexpected behavior.

❌ BAD:
```java
// Always disabling specific auto-configs without understanding impact
@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class, HibernateJpaAutoConfiguration.class})
public class MyApplication { /* ... */ }
```

✅ GOOD:
```java
// Only disable when you provide your own explicit configuration
// and the auto-config conflicts or is unwanted.
// For example, if you manage DataSource manually:
@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class})
public class MyApplication { /* ... */ }
```

## 5. Security Best Practices

### 5.1. Input Validation

Always validate user input at the API boundary. Use Spring's validation features.

❌ BAD:
```java
@PostMapping("/users")
public ResponseEntity<UserDto> createUser(@RequestBody UserDto userDto) {
    // No validation, directly use userDto
    userService.save(userDto);
    return ResponseEntity.ok(userDto);
}
```

✅ GOOD:
```java
@PostMapping("/users")
public ResponseEntity<UserDto> createUser(@Valid @RequestBody UserDto userDto) {
    // @Valid triggers validation based on JSR-303 annotations in UserDto
    userService.save(userDto);
    return ResponseEntity.ok(userDto);
}

// In UserDto:
public record UserDto(@NotNull Long id, @NotBlank @Size(min = 2, max = 50) String name) {}
```

## 6. Error Handling

### 6.1. Global Exception Handling

Use `@ControllerAdvice` for consistent global error handling.

❌ BAD:
```java
@RestController
public class UserController {
    // ...
    @GetMapping("/{id}")
    public UserDto getUser(@PathVariable Long id) {
        try {
            return userService.findById(id);
        } catch (UserNotFoundException e) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found", e);
        }
    }
}
```

✅ GOOD:
```java
@RestControllerAdvice // Global exception handler
public class GlobalExceptionHandler {
    @ExceptionHandler(UserNotFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ErrorResponse handleUserNotFound(UserNotFoundException ex) {
        return new ErrorResponse("USER_NOT_FOUND", ex.getMessage());
    }
}

// In UserController:
@RestController
public class UserController {
    // ...
    @GetMapping("/{id}")
    public UserDto getUser(@PathVariable Long id) {
        return userService.findById(id); // Let exception propagate
    }
}
// Custom exception:
public class UserNotFoundException extends RuntimeException { /* ... */ }
public record ErrorResponse(String code, String message) {}
```

## 7. API Design

### 7.1. RESTful Principles and DTOs

Design REST APIs following standard principles and use DTOs to decouple internal models from external API contracts.

❌ BAD:
```java
@RestController
@RequestMapping("/products")
public class ProductController {
    @Autowired private ProductService productService;

    @GetMapping("/{id}")
    public Product getProduct(@PathVariable Long id) { // Exposing internal entity
        return productService.findById(id);
    }
}
```

✅ GOOD:
```java
@RestController
@RequestMapping("/api/v1/products") // Versioned API path
public class ProductController {
    @Autowired private ProductService productService;

    @GetMapping("/{id}")
    public ProductResponseDto getProduct(@PathVariable Long id) {
        Product product = productService.findById(id);
        return ProductMapper.toDto(product); // Map to DTO
    }
}

public record ProductResponseDto(Long id, String name, BigDecimal price) {}
// ProductMapper is a utility to convert Product entity to ProductResponseDto
```

## 8. Testing Approaches

### 8.1. Layered Testing

Use specific Spring Boot test annotations for different layers.

❌ BAD:
```java
@SpringBootTest // Heavy for simple unit tests
public class UserServiceTest {
    @Autowired private UserService userService;
    // ...
}
```

✅ GOOD:
```java
// Unit test for service layer (mocking dependencies)
@ExtendWith(MockitoExtension.class)
public class UserServiceUnitTest {
    @Mock private UserRepository userRepository;
    @InjectMocks private UserService userService;

    @Test
    void findById_shouldReturnUser() {
        // ...
    }
}

// Integration test for web layer (no full context)
@WebMvcTest(UserController.class)
public class UserControllerWebMvcTest {
    @Autowired private MockMvc mockMvc;
    @MockBean private UserService userService; // Mock service dependency

    @Test
    void getUser_shouldReturnOk() throws Exception {
        // ...
    }
}

// Integration test for data layer
@DataJpaTest
public class UserRepositoryIntegrationTest {
    @Autowired private TestEntityManager entityManager;
    @Autowired private UserRepository userRepository;

    @Test
    void findById_shouldReturnProduct() {
        // ...
    }
}

// Full application context test (use sparingly)
@SpringBootTest
@AutoConfigureMockMvc
public class MyApplicationIntegrationTest {
    @Autowired private MockMvc mockMvc;
    // ...
}
```