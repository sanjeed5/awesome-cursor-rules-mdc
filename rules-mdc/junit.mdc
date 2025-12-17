---
description: This guide provides definitive, opinionated best practices for writing high-quality, maintainable, and effective unit tests using JUnit 5 (Jupiter) in Java projects.
globs: **/*.java
---
# junit Best Practices

JUnit 5 (Jupiter) is the bedrock of modern Java unit testing. This guide outlines the essential practices for crafting robust, readable, and reliable tests that integrate seamlessly into your development workflow. Adhere to these rules to ensure your tests are a valuable asset, not a burden.

## 1. Code Organization and Structure

### 1.1. Test File and Package Naming

Always mirror your production code's package structure within `src/test/java`. Name test classes by appending `Test` to the class they verify.

**❌ BAD:**
```java
// src/test/java/com/example/UserServiceLogic.java
package com.example;

class UserServiceLogic { /* ... */ }
```

**✅ GOOD:**
```java
// src/main/java/com/example/UserService.java
package com.example;
class UserService { /* ... */ }

// src/test/java/com/example/UserServiceTest.java
package com.example;
class UserServiceTest { /* ... */ }
```

### 1.2. Descriptive Test Method Naming

Test method names must clearly convey their purpose: the scenario being tested, the action, and the expected outcome. Use the `Given_When_Then` pattern or `methodName_condition_expectedResult`. For enhanced readability in test reports, leverage `@DisplayName`.

**❌ BAD:**
```java
@Test
void testAdd() { /* ... */ }

@Test
void calculateDiscount() { /* ... */ }
```

**✅ GOOD:**
```java
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    @DisplayName("Given two positive numbers, when added, then return their sum")
    void add_TwoPositiveNumbers_ReturnsSum() {
        // Arrange, Act, Assert
    }

    @Test
    void calculateDiscount_PremiumCustomer_Applies20PercentDiscount() {
        // Arrange, Act, Assert
    }
}
```

### 1.3. Logical Grouping with `@Nested`

For complex classes with many behaviors, use `@Nested` classes to group related tests. This improves readability and organization, especially when dealing with different states or conditions.

**❌ BAD:**
```java
class OrderServiceTest {
    @Test void createOrder_ValidInput_ReturnsOrder() { /* ... */ }
    @Test void createOrder_InvalidProduct_ThrowsException() { /* ... */ }
    @Test void cancelOrder_ExistingOrder_SetsStatusToCancelled() { /* ... */ }
    @Test void cancelOrder_NonExistingOrder_ThrowsException() { /* ... */ }
}
```

**✅ GOOD:**
```java
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class OrderServiceTest {

    @Nested
    @DisplayName("When creating an order")
    class CreateOrder {
        @Test
        void givenValidInput_thenReturnsOrder() { /* ... */ }

        @Test
        void givenInvalidProduct_thenThrowsException() { /* ... */ }
    }

    @Nested
    @DisplayName("When cancelling an order")
    class CancelOrder {
        @Test
        void givenExistingOrder_thenSetsStatusToCancelled() { /* ... */ }

        @Test
        void givenNonExistingOrder_thenThrowsException() { /* ... */ }
    }
}
```

## 2. Common Patterns and Anti-patterns

### 2.1. Follow Arrange-Act-Assert (AAA)

Structure every test method into three distinct sections:
1.  **Arrange**: Set up the test data, mocks, and the system under test.
2.  **Act**: Execute the method being tested.
3.  **Assert**: Verify the outcome using JUnit assertions.

This pattern enhances readability and makes it clear what each part of the test is doing.

**❌ BAD:**
```java
@Test
void processPayment_InsufficientFunds_ThrowsException() {
    BankAccount account = new BankAccount(100);
    PaymentProcessor processor = new PaymentProcessor();
    try {
        processor.process(account, 200);
        fail("Expected InsufficientFundsException");
    } catch (InsufficientFundsException e) {
        assertTrue(e.getMessage().contains("Insufficient funds"));
    }
}
```

**✅ GOOD:**
```java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PaymentProcessorTest {
    @Test
    void processPayment_InsufficientFunds_ThrowsException() {
        // Arrange
        BankAccount account = new BankAccount(100);
        PaymentProcessor processor = new PaymentProcessor();

        // Act & Assert
        InsufficientFundsException thrown = assertThrows(
            InsufficientFundsException.class,
            () -> processor.process(account, 200),
            "Expected process() to throw InsufficientFundsException"
        );
        assertTrue(thrown.getMessage().contains("Insufficient funds"));
    }
}
```

### 2.2. Single Logical Assertion Per Test

Each test should verify a single behavior. While you might use multiple `assert*` calls, they should collectively validate one logical outcome. This makes debugging easier: if a test fails, you know exactly which behavior is broken.

**❌ BAD:**
```java
@Test
void updateUser_ValidData_UpdatesUserAndLogsActivity() {
    User user = new User("old@example.com");
    userService.updateUser(user, "new@example.com");
    assertEquals("new@example.com", user.getEmail()); // Assertion 1
    assertTrue(activityLog.contains("User updated")); // Assertion 2 (different behavior)
}
```

**✅ GOOD:**
```java
// Test 1: Verify user update
@Test
void updateUser_ValidData_UpdatesUserEmail() {
    // Arrange
    User user = new User("old@example.com");
    // Act
    userService.updateUser(user, "new@example.com");
    // Assert
    assertEquals("new@example.com", user.getEmail());
}

// Test 2: Verify activity logging
@Test
void updateUser_ValidData_LogsUserActivity() {
    // Arrange
    User user = new User("old@example.com");
    // Act
    userService.updateUser(user, "new@example.com");
    // Assert
    assertTrue(activityLog.contains("User updated"));
}
```

### 2.3. Parameterized Tests for Data-Driven Scenarios

Avoid duplicating test logic for different inputs. Use `@ParameterizedTest` with `@ValueSource`, `@CsvSource`, or `@MethodSource` to run the same test logic with various data sets.

**❌ BAD:**
```java
@Test
void isEven_Two_ReturnsTrue() { assertTrue(calculator.isEven(2)); }
@Test
void isEven_Four_ReturnsTrue() { assertTrue(calculator.isEven(4)); }
@Test
void isEven_Odd_ReturnsFalse() { assertFalse(calculator.isEven(3)); }
```

**✅ GOOD:**
```java
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CalculatorTest {
    @ParameterizedTest(name = "{0} + {1} = {2}")
    @CsvSource({
        "1, 1, 2",
        "2, 3, 5",
        "10, 5, 15"
    })
    void add_VariousInputs_ReturnsCorrectSum(int a, int b, int expectedSum) {
        // Arrange
        Calculator calculator = new Calculator();
        // Act
        int actualSum = calculator.add(a, b);
        // Assert
        assertEquals(expectedSum, actualSum);
    }
}
```

## 3. Performance Considerations

### 3.1. Keep Tests Fast and Isolated

Unit tests must run quickly and deterministically. Avoid any operations that introduce slowness or flakiness.

**❌ BAD:**
```java
@Test
void processExternalService_Success_ReturnsData() {
    // Directly calls a slow external API or database
    ExternalService service = new ExternalService();
    String result = service.getData();
    assertNotNull(result);
}

@Test
void waitForAsyncOperation_Completes_ReturnsResult() throws InterruptedException {
    // Introduces non-deterministic delays
    Thread.sleep(1000);
    assertTrue(asyncService.isCompleted());
}
```

**✅ GOOD:**
```java
// Use mocks for external dependencies (see Mocking Strategies)
// Avoid Thread.sleep() or network calls in unit tests.
// If testing async, use Awaitility or similar libraries for integration tests,
// but for unit tests, mock the async behavior.
@Test
void processData_ValidInput_ReturnsProcessedData() {
    // Arrange
    DataProcessor processor = new DataProcessor(mockDependency); // Mocked dependency
    // Act
    String result = processor.process("input");
    // Assert
    assertEquals("processed_input", result);
}
```

### 3.2. Use `@Tag` for Filtering

Categorize tests by speed, type, or module using `@Tag`. This allows you to run subsets of tests (e.g., "fast" tests on every commit, "slow" integration tests less frequently).

```java
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class UserServiceTest {
    @Test
    @Tag("fast")
    void createUser_ValidUser_ReturnsCreatedUser() { /* ... */ }

    @Test
    @Tag("slow")
    @Tag("integration")
    void integrateWithDatabase_SavesUser_RetrievesUser() { /* ... */ }
}
```

## 4. Common Pitfalls and Gotchas

### 4.1. Avoid Logic Duplication in Tests

Never duplicate the logic you're testing within your test code. Hard-code expected results to ensure the test fails when the implementation is incorrect.

**❌ BAD:**
```java
@Test
void calculateArea_Circle_ReturnsCorrectArea() {
    double radius = 5.0;
    double expectedArea = Math.PI * radius * radius; // Duplicates calculation logic
    assertEquals(expectedArea, geometry.calculateCircleArea(radius));
}
```

**✅ GOOD:**
```java
@Test
void calculateArea_Circle_ReturnsCorrectArea() {
    // Arrange
    double radius = 5.0;
    // Act
    double actualArea = geometry.calculateCircleArea(radius);
    // Assert
    assertEquals(78.53981633974483, actualArea, 0.0001); // Hard-coded expected value
}
```

### 4.2. Flaky Tests (External Dependencies, System Date)

Tests must be deterministic. Avoid reliance on external services, databases, network calls, or system time/date. Mock these dependencies to ensure consistent results.

**❌ BAD:**
```java
@Test
void generateReport_CurrentDate_IncludesToday() {
    // Fails if run on a different day or timezone
    String report = reportGenerator.generate();
    assertTrue(report.contains(LocalDate.now().toString()));
}
```

**✅ GOOD:**
```java
// Use a clock abstraction that can be mocked
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReportGeneratorTest {
    @Test
    void generateReport_FixedDate_IncludesExpectedDate() {
        // Arrange
        Clock fixedClock = Clock.fixed(Instant.parse("2025-01-15T10:00:00Z"), ZoneId.of("UTC"));
        ReportGenerator reportGenerator = new ReportGenerator(fixedClock); // Inject mockable clock
        // Act
        String report = reportGenerator.generate();
        // Assert
        assertTrue(report.contains("2025-01-15"));
    }
}
```

## 5. Mocking Strategies

### 5.1. Replace Heavyweight Dependencies with Mocks

Use a mocking framework (e.g., Mockito) to isolate the unit under test from its dependencies. This keeps tests fast, focused, and prevents side effects.

**❌ BAD:**
```java
// Directly instantiates a real, slow, or stateful dependency
class UserService {
    private final DatabaseRepository repository = new DatabaseRepository();
    // ...
}

@Test
void createUser_ValidUser_SavesToDatabase() {
    UserService userService = new UserService();
    User user = new User("test@example.com");
    userService.createUser(user);
    // This test now hits a real database!
    User savedUser = retrieveFromRealDatabase(user.getEmail());
    assertNotNull(savedUser);
}
```

**✅ GOOD:**
```java
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class UserServiceTest {
    @Mock
    private DatabaseRepository databaseRepository; // Mock the dependency

    @InjectMocks
    private UserService userService; // Inject mocks into the service under test

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this); // Initialize mocks
    }

    @Test
    void createUser_ValidUser_SavesUserViaRepository() {
        // Arrange
        User user = new User("test@example.com");
        when(databaseRepository.save(any(User.class))).thenReturn(user); // Define mock behavior

        // Act
        User createdUser = userService.createUser(user);

        // Assert
        assertNotNull(createdUser);
        verify(databaseRepository, times(1)).save(user); // Verify interaction with mock
    }
}
```

## 6. Coverage Patterns

### 6.1. Focus on Meaningful Coverage

Aim for high, but meaningful, code coverage. Don't chase 100% coverage with superficial tests. Prioritize testing critical paths, boundary conditions, and error handling.

**❌ BAD:**
```java
// Test only happy path, ignores edge cases and exceptions
@Test
void divide_PositiveNumbers_ReturnsQuotient() {
    assertEquals(2, calculator.divide(4, 2));
}
```

**✅ GOOD:**
```java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CalculatorTest {
    @Test
    void divide_PositiveNumbers_ReturnsQuotient() {
        assertEquals(2, calculator.divide(4, 2));
    }

    @Test
    void divide_ByZero_ThrowsArithmeticException() {
        assertThrows(ArithmeticException.class, () -> calculator.divide(10, 0));
    }

    @Test
    void divide_NegativeNumbers_ReturnsCorrectQuotient() {
        assertEquals(-2, calculator.divide(-4, 2));
    }
}
```