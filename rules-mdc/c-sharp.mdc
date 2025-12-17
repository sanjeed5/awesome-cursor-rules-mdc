---
description: This guide defines definitive C# coding standards and best practices for our team, covering naming, formatting, modern language features, performance, security, and error handling to ensure consistent, maintainable, and high-quality code.
globs: **/*
---
# c-sharp Best Practices

This document outlines the definitive C# coding standards and best practices for our team. Adhering to these guidelines ensures consistency, readability, maintainability, and high performance across all our C# projects.

## Code Organization and Structure

### 1. Naming Conventions
Always follow standard .NET naming conventions to improve readability and predictability.

*   **Types (Classes, Structs, Enums, Delegates, Records):** `PascalCase`
*   **Interfaces:** `IPascalCase`
*   **Methods, Public Properties, Events:** `PascalCase`
*   **Public Constants:** `PascalCase`
*   **Local Variables, Method Parameters:** `camelCase`
*   **Private/Protected Fields:** `_camelCase` (underscore prefix)
*   **Async Methods:** `PascalCase` ending with `Async`

❌ BAD:
```csharp
public class myService {
    private string name;
    public void getdata() { /* ... */ }
    public async Task do_work() { /* ... */ }
    public interface IMyServiceImplementation { /* ... */ } // Incorrect interface naming
}
```

✅ GOOD:
```csharp
public class MyService {
    private readonly ILogger _logger; // Readonly private fields
    private string _name;
    public const int MaxRetries = 3; // Public constants are PascalCase

    public MyService(ILogger logger) {
        _logger = logger;
    }

    public void GetData() { /* ... */ }
    public async Task DoWorkAsync() { /* ... */ }
}
public interface IService { /* ... */ } // Correct interface naming
```

### 2. Formatting
Always maintain consistent code layout for clarity.

*   **Indentation:** Use **4 spaces**. Configure your IDE to use spaces, not tabs.
*   **Braces:** Use [Allman style](https://en.wikipedia.org/wiki/Indent_style#Allman_style) where each brace begins on a new line. Always use braces, even for single-statement blocks.
*   **`using` Directives:** Place at the top of the file, outside the namespace. Sort alphabetically, with `System` namespaces first.
*   **Auto-properties:** Keep on a single line.
*   **Generic Constraints:** Place on separate, indented lines.

❌ BAD:
```csharp
using System.Collections.Generic;
using System;
namespace MyNamespace {
class MyClass {
    public int Age {get;set;}
    void DoSomething() {
        if(true)
            Console.WriteLine("Hello");
    }
}}
```

✅ GOOD:
```csharp
using System;
using System.Collections.Generic;

namespace MyNamespace
{
    public class MyClass
    {
        public int Age { get; set; }

        public void DoSomething()
        {
            if (true)
            {
                Console.WriteLine("Hello");
            }
        }
    }
}
```

### 3. Modern Language Features
Always embrace modern C# features for cleaner, more concise, and safer code.

*   **`var` Keyword:** Use `var` when the type is obvious from the right-hand side of the assignment. Avoid `var` when the type is not immediately clear.
*   **Expression-bodied Members:** Use for single-expression methods, properties, and accessors.
*   **Object/Collection Initializers:** Prefer for creating and populating objects/collections.
*   **Pattern Matching:** Use for type checks and value comparisons.
*   **Nullable Reference Types:** Enable and address warnings (`#nullable enable`).
*   **Records:** Use for immutable data transfer objects (DTOs) or value-based types.

❌ BAD:
```csharp
string name = "John Doe";
public int GetCount() { return _items.Count; }
MyObject obj = new MyObject();
obj.Property1 = "Value1";
obj.Property2 = 123;
if (item is string) { Console.WriteLine((string)item); }
```

✅ GOOD:
```csharp
var name = "John Doe"; // Type is obvious
public int Count => _items.Count; // Expression-bodied property
public void Log(string message) => _logger.LogInformation(message); // Expression-bodied method

var obj = new MyObject {
    Property1 = "Value1",
    Property2 = 123
};

if (item is string s) { Console.WriteLine(s); } // Pattern matching
```

## Common Patterns and Anti-patterns

### 1. Dependency Injection (DI)
Always use constructor injection for dependencies. Never use service locators or direct instantiation of dependencies within classes.

❌ BAD:
```csharp
public class MyService
{
    private readonly IRepository _repository = new ConcreteRepository(); // Direct instantiation
    // Or: private readonly IRepository _repository = ServiceLocator.Resolve<IRepository>();
}
```

✅ GOOD:
```csharp
public class MyService
{
    private readonly IRepository _repository;

    public MyService(IRepository repository) // Constructor injection
    {
        _repository = repository;
    }
}
```

### 2. Immutability
Favor immutable objects, especially for DTOs and shared state, to reduce side effects and improve thread safety. Use `record` types or `init` accessors.

❌ BAD:
```csharp
public class User
{
    public string FirstName { get; set; }
    public string LastName { get; set; }
}
```

✅ GOOD:
```csharp
public record User(string FirstName, string LastName); // Immutable record

// Or for classes:
public class Product
{
    public int Id { get; init; } // Init-only setter
    public string Name { get; init; }
}
```

## Performance Considerations

### 1. Asynchronous Programming
Always use `async/await` for I/O-bound operations to prevent blocking threads. Avoid `async void` methods (except for event handlers) and blocking `async` calls.

❌ BAD:
```csharp
public void DoWork() {
    SomeAsyncOperation().Wait(); // Blocks the thread
}
public async void Event_Click(object sender, EventArgs e) { /* ... */ } // Async void outside event handlers
```

✅ GOOD:
```csharp
public async Task DoWorkAsync() {
    await SomeAsyncOperation(); // Awaits without blocking
}
// For event handlers, async void is acceptable
public async void Button_Click(object sender, EventArgs e) {
    await LoadDataAsync();
}
```

### 2. String Manipulation
Always use `StringBuilder` for concatenating many strings in loops.

❌ BAD:
```csharp
string result = "";
for (int i = 0; i < 1000; i++) {
    result += i.ToString(); // Inefficient in a loop, creates many new strings
}
```

✅ GOOD:
```csharp
var sb = new System.Text.StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.Append(i);
}
string result = sb.ToString();
```

## Common Pitfalls and Gotchas

### 1. Magic Strings/Numbers
Never hardcode string literals or numeric values directly in code. Always use constants or configuration.

❌ BAD:
```csharp
public void ProcessOrder(string status) {
    if (status == "Pending") { /* ... */ }
}
public decimal CalculateDiscount(decimal price) {
    return price * 0.10m;
}
```

✅ GOOD:
```csharp
public static class OrderStatus {
    public const string Pending = "Pending";
    public const string Completed = "Completed";
}
public const decimal DiscountRate = 0.10m;

public void ProcessOrder(string status) {
    if (status == OrderStatus.Pending) { /* ... */ }
}
public decimal CalculateDiscount(decimal price) {
    return price * DiscountRate;
}
```

### 2. Deep Nesting
Always refactor deeply nested `if`/`for` blocks to improve readability. Use guard clauses or extract methods.

❌ BAD:
```csharp
if (condition1) {
    if (condition2) {
        if (condition3) {
            // ... complex logic
        }
    }
}
```

✅ GOOD:
```csharp
if (!condition1) return;
if (!condition2) return;
if (!condition3) return;
// ... simplified logic
```

## Security Best Practices

### 1. Input Validation
Always validate all external input (user input, API calls, file contents) at the boundaries of your application.

❌ BAD:
```csharp
public void CreateUser(string username, string password) {
    // Directly use inputs without validation
    _userRepository.Add(new User(username, password));
}
```

✅ GOOD:
```csharp
public void CreateUser(string username, string password) {
    if (string.IsNullOrWhiteSpace(username) || username.Length < 5) {
        throw new ArgumentException("Invalid username.");
    }
    // ... more validation (e.g., password strength, SQL injection prevention)
    _userRepository.Add(new User(username, password));
}
```

### 2. Secure Data Handling
Never store sensitive information (passwords, API keys) in plain text. Always use secure configuration, environment variables, or secrets management solutions.

## Error Handling

### 1. Specific Exceptions
Always catch specific exceptions rather than general `Exception`. Log and rethrow when appropriate.

❌ BAD:
```csharp
try {
    // ...
} catch (Exception ex) { // Too broad, hides specific issues
    _logger.LogError(ex, "An error occurred.");
    throw; // Rethrows original exception, losing stack trace if not careful
}
```

✅ GOOD:
```csharp
try {
    // ...
} catch (FileNotFoundException ex) {
    _logger.LogError(ex, "File not found at path {Path}.", _filePath);
    // Specific handling, e.g., notify user
} catch (IOException ex) {
    _logger.LogError(ex, "An I/O error occurred during file operation.");
    throw; // Rethrow if caller needs to handle, preserving stack trace
}
```

### 2. `using` for `IDisposable`
Always use the `using` statement for objects that implement `IDisposable` to ensure proper resource cleanup.

❌ BAD:
```csharp
SqlConnection connection = new SqlConnection("...");
connection.Open();
// ... forgot to close/dispose
```

✅ GOOD:
```csharp
using (var connection = new SqlConnection("..."))
{
    connection.Open();
    // ...
} // Connection is automatically disposed here
```

## API Design

### 1. Clear Contracts
Always define clear, well-documented API contracts using DTOs and interfaces. Avoid exposing internal implementation details.

### 2. Asynchronous APIs
Always expose `async` methods for any I/O-bound operations in public APIs.

❌ BAD:
```csharp
public IEnumerable<Product> GetProducts() { /* Sync I/O operations */ }
```

✅ GOOD:
```csharp
public Task<IEnumerable<Product>> GetProductsAsync() { /* Async I/O operations */ }
```

## Testing Approaches

### 1. Unit Testing
Always write focused unit tests for individual components, mocking external dependencies.

### 2. Dependency Injection for Testability
Always design classes with Dependency Injection to make them easily testable.

❌ BAD:
```csharp
public class OrderProcessor {
    private readonly PaymentGateway _gateway = new PaymentGateway(); // Hard dependency
    public void Process(Order order) { _gateway.Charge(order.Amount); }
}
```

✅ GOOD:
```csharp
public interface IPaymentGateway { void Charge(decimal amount); }
public class PaymentGateway : IPaymentGateway { /* ... */ }

public class OrderProcessor {
    private readonly IPaymentGateway _gateway;
    public OrderProcessor(IPaymentGateway gateway) { _gateway = gateway; }
    public void Process(Order order) { _gateway.Charge(order.Amount); }
}
// Now IPaymentGateway can be easily mocked in tests.
```