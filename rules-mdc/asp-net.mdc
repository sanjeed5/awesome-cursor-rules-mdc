---
description: This guide provides opinionated, actionable best practices for building high-performance, maintainable, and secure ASP.NET Core 9+ applications, focusing on modern C# patterns and common pitfalls.
globs: **/*
---
# asp-net Best Practices

This document outlines the definitive best practices for developing ASP.NET Core applications within our team. Adhere to these guidelines to ensure consistent, performant, and maintainable backend services.

## 1. Code Organization and Structure

Organize your projects by feature or bounded context, not by technical layer. This improves cohesion and reduces coupling.

### ✅ GOOD: Feature-First Organization

```csharp
// Project structure for a "Product" feature
src/
├── MyProject.Api/
│   ├── Program.cs
│   ├── Features/
│   │   ├── Products/
│   │   │   ├── ProductEndpoints.cs // Minimal API definitions
│   │   │   ├── ProductController.cs // API Controller (if used)
│   │   │   ├── GetProduct.cs      // Request/Response DTOs, Handler (e.g., MediatR)
│   │   │   ├── CreateProduct.cs
│   │   │   └── ProductService.cs  // Feature-specific business logic
│   ├── Common/                  // Shared infrastructure, middleware, extensions
│   │   ├── Filters/
│   │   ├── Middleware/
│   │   └── Extensions/
├── MyProject.Application/       // Application-specific services, interfaces, DTOs
│   ├── Interfaces/
│   ├── Services/
│   └── DTOs/
├── MyProject.Domain/            // Core domain entities, value objects, interfaces
├── MyProject.Infrastructure/    // Data access (EF Core), external services
│   ├── Data/
│   ├── Repositories/
│   └── ExternalServices/
└── MyProject.Tests/             // Unit and Integration tests
```

### ❌ BAD: Layer-First Organization

```csharp
// Avoid this structure for larger applications
src/
├── MyProject.Api/
│   ├── Controllers/ // All controllers here
│   ├── Services/    // All services here
│   ├── Repositories/ // All repositories here
│   └── DTOs/        // All DTOs here
```

## 2. API Design: Minimal APIs vs. API Controllers

**Always prefer Minimal APIs** for new endpoints due to their simplicity and directness. Use API Controllers only when you require advanced features like automatic model binding from multiple sources, built-in validation attributes, or complex attribute routing that Minimal APIs don't easily provide.

### ✅ GOOD: Minimal API for Simplicity

```csharp
// Program.cs or dedicated endpoint file
app.MapGet("/products/{id}", async (Guid id, IProductService service) =>
{
    var product = await service.GetProductByIdAsync(id);
    return product is not null ? Results.Ok(product) : Results.NotFound();
})
.WithName("GetProductById")
.Produces<ProductDto>(StatusCodes.Status200OK)
.Produces(StatusCodes.Status404NotFound);
```

### ✅ GOOD: API Controller for Complex Scenarios

```csharp
[ApiController]
[Route("api/[controller]")]
public class OrdersController : ControllerBase
{
    private readonly IOrderService _orderService;

    public OrdersController(IOrderService orderService) => _orderService = orderService;

    [HttpPost]
    [ProducesResponseType(StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> CreateOrder([FromBody] CreateOrderRequest request)
    {
        if (!ModelState.IsValid) return BadRequest(ModelState);
        var order = await _orderService.CreateOrderAsync(request);
        return CreatedAtAction(nameof(GetOrderById), new { id = order.Id }, order);
    }
}
```

## 3. Performance Considerations

### 3.1. Avoid Blocking Calls

**Always use `async/await`** for I/O-bound operations (database, network, file system). Never block the thread pool.

### ❌ BAD: Blocking Calls

```csharp
// Synchronous I/O in an async context
public Product GetProduct(Guid id)
{
    // This blocks the calling thread while waiting for the DB
    return _dbContext.Products.Find(id);
}

// Blocking on an async task
var result = SomeAsyncTask().Result; // Blocks!
SomeAsyncTask().Wait();             // Blocks!
```

### ✅ GOOD: Asynchronous I/O

```csharp
public async Task<Product?> GetProductAsync(Guid id)
{
    return await _dbContext.Products.FindAsync(id);
}

// Correctly awaiting an async task
var result = await SomeAsyncTask();
```

### 3.2. Return Paged Collections

**Always paginate large collections** to prevent `OutOfMemoryException` and slow responses.

### ❌ BAD: Returning Large Collections Unpaged

```csharp
app.MapGet("/products", async (IProductService service) =>
{
    var allProducts = await service.GetAllProductsAsync(); // Potentially huge
    return Results.Ok(allProducts);
});
```

### ✅ GOOD: Paging Collections

```csharp
app.MapGet("/products", async (int page = 1, int pageSize = 20, IProductService service) =>
{
    var pagedProducts = await service.GetPagedProductsAsync(page, pageSize);
    return Results.Ok(pagedProducts);
});
```

### 3.3. Minimize Large Object Allocations

Cache frequently used large objects. Avoid creating large objects (>= 85KB) in hot code paths to reduce GC pressure. Use `ArrayPool<T>` for large arrays.

## 4. Dependency Injection (DI)

**Always use ASP.NET Core's built-in DI container.** Register services with the correct lifetime.

*   **Singleton**: For stateless services, configuration objects, or services that manage shared state.
*   **Scoped**: For services tied to a single HTTP request (e.g., `DbContext`, `IUnitOfWork`).
*   **Transient**: For lightweight services that should be new for every injection (rarely needed for typical backend services).

### ✅ GOOD: Correct Service Lifetimes

```csharp
public void ConfigureServices(IServiceCollection services)
{
    // Caching service can be shared across the app
    services.AddSingleton<ICacheService, RedisCacheService>();

    // Database context is scoped to the request
    services.AddDbContext<AppDbContext>(options => /* ... */, ServiceLifetime.Scoped);
    services.AddScoped<IProductRepository, ProductRepository>();

    // Business logic service, typically scoped
    services.AddScoped<IProductService, ProductService>();
}
```

## 5. Security Best Practices

### 5.1. Authentication and Authorization

**Always secure your API endpoints.** Use JWT Bearer tokens for API authentication. Define and enforce authorization policies.

### ✅ GOOD: Policy-Based Authorization

```csharp
// Program.cs
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options => { /* ... configure JWT ... */ });
builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("CanManageProducts", policy =>
        policy.RequireRole("Admin", "ProductManager"));
});

// Minimal API endpoint
app.MapPost("/products", async (CreateProductRequest request, IProductService service) =>
{
    // ...
}).RequireAuthorization("CanManageProducts");

// Controller action
[Authorize(Policy = "CanManageProducts")]
[HttpPost]
public async Task<IActionResult> CreateProduct([FromBody] CreateProductRequest request)
{ /* ... */ }
```

### 5.2. Input Validation

**Always validate all incoming request data.** Use Data Annotations or FluentValidation for robust validation.

### ✅ GOOD: DTO Validation

```csharp
public record CreateProductRequest(
    [Required] [StringLength(100)] string Name,
    [Range(0.01, 10000.00)] decimal Price,
    [Url] string? ImageUrl
);

// In Minimal API or Controller, validation is often automatic with [ApiController]
// For Minimal APIs, you might explicitly check if not using a framework that handles it:
app.MapPost("/products", (CreateProductRequest request) =>
{
    var validationContext = new ValidationContext(request);
    var validationResults = new List<ValidationResult>();
    if (!Validator.TryValidateObject(request, validationContext, validationResults, true))
    {
        return Results.ValidationProblem(validationResults.ToDictionary(r => r.MemberNames.First(), r => new[] { r.ErrorMessage }));
    }
    // ... process valid request
});
```

## 6. Error Handling

**Implement global exception handling middleware.** Provide consistent, developer-friendly error responses without exposing sensitive details.

### ✅ GOOD: Global Exception Handler

```csharp
// Program.cs
app.UseExceptionHandler(appBuilder =>
{
    appBuilder.Run(async context =>
    {
        var exceptionHandlerPathFeature = context.Features.Get<IExceptionHandlerPathFeature>();
        var exception = exceptionHandlerPathFeature?.Error;

        var problemDetails = new ProblemDetails
        {
            Status = StatusCodes.Status500InternalServerError,
            Title = "An error occurred while processing your request.",
            Detail = "Please try again later. If the problem persists, contact support.",
            Instance = context.Request.Path
        };

        // Log the exception details for internal monitoring
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
        logger.LogError(exception, "Unhandled exception for request {Path}", context.Request.Path);

        context.Response.StatusCode = problemDetails.Status.Value;
        context.Response.ContentType = "application/problem+json";
        await context.Response.WriteAsJsonAsync(problemDetails);
    });
});
```

## 7. Structured Logging

**Always use structured logging** to enable easy querying and analysis of logs. Prefer `ILogger<T>` for context-specific logging.

### ✅ GOOD: Structured Logging

```csharp
public class ProductService : IProductService
{
    private readonly ILogger<ProductService> _logger;

    public ProductService(ILogger<ProductService> logger) => _logger = logger;

    public async Task<ProductDto> CreateProductAsync(CreateProductRequest request)
    {
        _logger.LogInformation("Creating product {ProductName} with price {ProductPrice}", request.Name, request.Price);
        // ...
        _logger.LogInformation("Product {ProductId} created successfully.", newProduct.Id);
        return newProductDto;
    }
}
```

### ❌ BAD: Unstructured Logging

```csharp
_logger.LogInformation("Creating product with name: " + request.Name + " and price: " + request.Price);
```

## 8. Testing Approaches

**Prioritize unit and integration tests.**

*   **Unit Tests**: Focus on isolated business logic (services, domain models). Mock all external dependencies.
*   **Integration Tests**: Verify API endpoints, middleware, and database interactions. Use `WebApplicationFactory<TStartup>` for in-memory testing.

### ✅ GOOD: Unit Test Example

```csharp
public class ProductServiceTests
{
    [Fact]
    public async Task CreateProductAsync_ValidRequest_ReturnsProductDto()
    {
        // Arrange
        var mockRepo = new Mock<IProductRepository>();
        mockRepo.Setup(r => r.AddAsync(It.IsAny<Product>())).ReturnsAsync(new Product { Id = Guid.NewGuid(), Name = "Test", Price = 10m });
        var service = new ProductService(mockRepo.Object, Mock.Of<ILogger<ProductService>>());
        var request = new CreateProductRequest("Test Product", 10.00m, null);

        // Act
        var result = await service.CreateProductAsync(request);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Test Product", result.Name);
        mockRepo.Verify(r => r.AddAsync(It.IsAny<Product>()), Times.Once);
    }
}
```

### ✅ GOOD: Integration Test Example

```csharp
public class ProductsApiTests : IClassFixture<CustomWebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ProductsApiTests(CustomWebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetProducts_ReturnsSuccessAndCorrectContentType()
    {
        // Act
        var response = await _client.GetAsync("/products");

        // Assert
        response.EnsureSuccessStatusCode(); // Status Code 200-299
        Assert.Equal("application/json; charset=utf-8", response.Content.Headers.ContentType?.ToString());
    }
}
```