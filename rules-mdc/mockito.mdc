---
description: This guide provides definitive best practices for writing robust, maintainable, and readable unit tests using Mockito with JUnit 5, focusing on modern patterns and common pitfalls.
globs: **/*.java
---
# Mockito Best Practices

Mockito is the indispensable mocking framework for Java unit testing. Adhere to these guidelines to ensure your tests are clear, reliable, and resistant to refactoring.

## 1. Standardize Test Setup with JUnit 5 & Annotations

Always use JUnit 5's `@ExtendWith` for Mockito integration and leverage annotations for mock initialization. This keeps your setup concise and automatic.

❌ **BAD: Manual setup or old JUnit 4 runner**
```java
// JUnit 4: @RunWith(MockitoJUnitRunner.class) - Obsolete
// Manual mock creation
UserRepository repo = mock(UserRepository.class);
UserService service = new UserService(repo);
```

✅ **GOOD: JUnit 5 `@ExtendWith` and annotations**
```java
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.InjectMocks;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    @Mock
    UserRepository userRepository; // Collaborator

    @InjectMocks
    UserService userService; // System Under Test (SUT)

    // ... tests
}
```

## 2. One Test, One Behavior

Each test method must verify a single, distinct behavior. This makes tests easier to debug and provides clear documentation.

❌ **BAD: Multiple unrelated assertions in one test**
```java
@Test
void processOrder_createsUser_addsPoints_sendsEmail() {
    // ... setup
    orderService.placeOrder(orderRequest);
    verify(userRepository).save(any(User.class));
    verify(loyaltyService).addPoints(anyString(), anyInt());
    verify(notificationService).sendEmail(anyString(), anyString());
}
```

✅ **GOOD: Focused tests**
```java
@Test
void shouldSaveUserWhenOrderIsPlaced() {
    orderService.placeOrder(orderRequest);
    verify(userRepository).save(any(User.class));
}

@Test
void shouldAddLoyaltyPointsForEligibleUsers() {
    when(loyaltyPolicy.isEligible(any())).thenReturn(true);
    orderService.placeOrder(orderRequest);
    verify(loyaltyService).addPoints(anyString(), anyInt());
}
```

## 3. Verify Outcomes, Not Implementation Details

Only verify interactions that are part of the contract being tested. Avoid verifying internal method calls or private state.

❌ **BAD: Verifying internal helper method calls**
```java
// Assuming OrderService.placeOrder() internally calls calculateTax()
@Test
void shouldCalculateTaxWhenPlacingOrder() {
    orderService.placeOrder(orderRequest);
    verify(taxCalculator).calculateTax(any()); // Tax calculation is an internal detail
}
```

✅ **GOOD: Verifying the observable outcome**
```java
@Test
void shouldReturnOrderWithCorrectTotalIncludingTax() {
    when(taxCalculator.calculateTax(any())).thenReturn(BigDecimal.valueOf(10));
    Order result = orderService.placeOrder(orderRequest);
    assertEquals(BigDecimal.valueOf(110), result.getTotalAmount()); // Verify the final amount
}
```

## 4. Handle `UnnecessaryStubbingException` Correctly

Mockito's strict stubbing (default with `MockitoExtension`) catches unused stubs. This prevents dead code in tests. Resolve it by removing unused stubs. Use `lenient()` only when absolutely necessary (e.g., in parameterized tests where some stubs might not be hit in all cases).

❌ **BAD: Silencing all strictness with `@MockitoSettings(strictness = Strictness.LENIENT)`**
```java
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT) // Hides potential issues
class MyServiceTest {
    // ...
    @Test
    void someTest() {
        when(mockDependency.someMethod()).thenReturn("value"); // This stub is never used
        // ... test logic that doesn't call someMethod()
    }
}
```

✅ **GOOD: Remove unused stubs or use `lenient()` for specific stubs**
```java
@ExtendWith(MockitoExtension.class) // Strictness is default
class MyServiceTest {
    // ...
    @Test
    void someTest() {
        // No unused stubs here
        // If a stub is conditionally used, make it lenient:
        Mockito.lenient().when(mockDependency.optionalMethod()).thenReturn("optional");
        // ...
    }
}
```

## 5. Use `ArgumentCaptor` for Complex Argument Verification

When you need to inspect the exact values of objects passed to a mock, `ArgumentCaptor` is your tool.

❌ **BAD: Generic `any()` when specific content matters**
```java
verify(invoiceRepository).save(any(Invoice.class)); // Only checks type
```

✅ **GOOD: Capture and assert on the actual object's state**
```java
import org.mockito.ArgumentCaptor;
// ...
@Test
void shouldSaveInvoiceWithCorrectDetails() {
    // ... call SUT method
    ArgumentCaptor<Invoice> captor = ArgumentCaptor.forClass(Invoice.class);
    verify(invoiceRepository).save(captor.capture());
    Invoice savedInvoice = captor.getValue();
    assertEquals("ORD-123", savedInvoice.getOrderId());
    assertEquals(BigDecimal.valueOf(150.00), savedInvoice.getAmount());
}
```

## 6. Prefer Mocks over Spies

Use mocks for external, unmanaged dependencies where you control all behavior. Use spies only when you need partial real behavior of an object.

❌ **BAD: Spying on simple dependencies that should be fully mocked**
```java
@Spy
UserService userService = new UserService(userRepository); // Unnecessary complexity
```

✅ **GOOD: Mocking dependencies, using `@InjectMocks` for SUT**
```java
@Mock
UserRepository userRepository;

@InjectMocks
UserService userService; // SUT is a real instance, dependencies are mocked
```

## 7. Use Parameterized Tests for Similar Scenarios

Avoid duplicate test methods for different inputs. Parameterized tests keep your test suite DRY and maintainable.

❌ **BAD: Repetitive tests for different inputs**
```java
@Test void shouldReturnBronzeFor0Points() { /* ... */ }
@Test void shouldReturnSilverFor1000Points() { /* ... */ }
```

✅ **GOOD: Single parameterized test**
```java
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
// ...
class LoyaltyServiceTest {
    @InjectMocks
    LoyaltyService loyaltyService;

    @ParameterizedTest
    @CsvSource({
        "0, BRONZE",
        "1000, SILVER",
        "5000, GOLD"
    })
    void shouldReturnCorrectTierForPoints(int points, String expectedTier) {
        String actualTier = loyaltyService.calculateTier(points);
        assertEquals(expectedTier, actualTier);
    }
}
```

## 8. Mock Final Classes, Static Methods, and Constructors (with Caution)

Mockito 5 (and newer) supports mocking these without extra dependencies. Use this feature judiciously for testing legacy code or third-party APIs that were not designed for testability. Avoid it for your own code; design for testability instead.

```java
// Example (use sparingly for legacy/third-party code)
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;
import org.mockito.MockedStatic;

@Test
void shouldMockStaticMethod() {
    try (MockedStatic<MyStaticUtil> mockedStatic = mockStatic(MyStaticUtil.class)) {
        when(MyStaticUtil.getConstantValue()).thenReturn("MOCKED_VALUE");
        assertEquals("MOCKED_VALUE", MyStaticUtil.getConstantValue());
    }
}
```