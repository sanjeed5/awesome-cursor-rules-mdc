---
description: This guide defines definitive best practices for structuring, scripting, and testing APIs using Postman, ensuring maintainable, robust, and collaborative API workflows.
globs: **/*
---
# Postman Best Practices

Postman is our definitive tool for API development, testing, and documentation. Adhering to these guidelines ensures consistency, maintainability, and reliability across all our API interactions.

## 1. Code Organization and Structure

Organize your Postman collections like source code: structured, versioned, and modular.

### 1.1 Collection Hierarchy

Mirror your API's resource structure with a clear folder hierarchy. Each folder should represent a logical API resource or module.

❌ **BAD: Flat Collection**
```
- My API
  - Create User
  - Get User by ID
  - Update Product
  - Delete Order
```

✅ **GOOD: Structured Collection**
```
- My API v1.0
  - Users
    - POST Create User
    - GET Get User by ID
    - PUT Update User
    - DELETE Delete User
  - Products
    - POST Create Product
    - GET Get Product by ID
    - PUT Update Product
    - DELETE Delete Product
  - Orders
    - POST Create Order
    - GET Get Order by ID
    - PUT Update Order
    - DELETE Delete Order
```

### 1.2 Request Naming

Use descriptive names for requests, including the HTTP method and the resource path.

❌ **BAD: Vague Request Names**
```
- Get User
- Update Product
```

✅ **GOOD: Explicit Request Names**
```
- GET /users/{id}
- PUT /products/{id}
- POST /auth/login
```

### 1.3 Variables Management

Leverage Postman variables (global, environment, collection) to avoid hardcoding and promote reusability.

*   **Global Variables**: For non-sensitive, truly global data (e.g., common utility functions). Use sparingly.
*   **Environment Variables**: For environment-specific configurations (e.g., `BASE_URL`, `AUTH_TOKEN`, `API_KEY`). **Never commit sensitive data to source control.**
*   **Collection Variables**: For data specific to a collection (e.g., `user_id`, `product_name`) that might change during a test run but is not environment-specific.

**Naming Convention**:
*   Environment Variables: `UPPER_SNAKE_CASE` (e.g., `DEV_BASE_URL`, `PROD_AUTH_TOKEN`)
*   Collection/Local Variables: `camelCase` (e.g., `userId`, `productName`)

❌ **BAD: Hardcoded Values**
```javascript
// Pre-request Script
pm.request.url = "https://dev.api.com/users";
pm.request.headers.add({ key: "Authorization", value: "Bearer my-hardcoded-token" });
```

✅ **GOOD: Variable Usage**
```javascript
// Environment Variables: BASE_URL, authToken
// Collection Variables: userId

// Pre-request Script
pm.request.url = `${pm.environment.get("BASE_URL")}/users/${pm.collection.get("userId")}`;
pm.request.headers.add({ key: "Authorization", value: `Bearer ${pm.environment.get("authToken")}` });
```

## 2. Common Patterns and Anti-patterns

Adopt patterns that enhance test reliability and maintainability.

### 2.1 Reusable Pre-request and Test Scripts

Extract shared logic into collection-level or folder-level pre-request/test scripts. This reduces duplication and centralizes updates.

❌ **BAD: Duplicated Test Logic**
```javascript
// Request A Test Script
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
// Request B Test Script (same logic)
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

✅ **GOOD: Reusable Collection-level Test Script**
```javascript
// Collection-level Test Script
pm.test("Response time is less than 200ms", function () {
    pm.expect(pm.response.responseTime).to.be.below(200);
});

// For specific requests, add request-level tests
// Request A Test Script
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});
```

### 2.2 Chaining Requests

Pass data between requests using environment or collection variables. This is crucial for end-to-end flows.

```javascript
// Request 1: POST /users (creates a user)
// Test Script
pm.test("User created successfully", function () {
    pm.response.to.have.status(201);
    const response = pm.response.json();
    pm.environment.set("newUserId", response.id); // Store ID for subsequent requests
});

// Request 2: GET /users/{{newUserId}} (retrieves the created user)
// Pre-request Script (if needed)
// No specific pre-request needed here, as {{newUserId}} is used directly in URL.
```

### 2.3 Data-Driven Testing

Utilize the Collection Runner with external CSV or JSON data files for testing multiple scenarios or large datasets.

```json
// data.json
[
  { "username": "testuser1", "password": "password1" },
  { "username": "testuser2", "password": "password2" }
]
```
```javascript
// Request: POST /auth/login
// Request Body:
// {
//   "username": "{{username}}",
//   "password": "{{password}}"
// }

// Test Script
pm.test("Login successful for {{username}}", function () {
    pm.response.to.have.status(200);
    pm.expect(pm.response.json().token).to.be.a('string');
});
```
*Run this request via Collection Runner, selecting `data.json` as the data file.*

## 3. Performance Considerations

Optimize your test suite for speed and reliability.

### 3.1 Mock Servers for External Dependencies

Use Postman Mock Servers to simulate third-party APIs during end-to-end testing. This makes tests faster, deterministic, and independent of external service availability.

❌ **BAD: Relying on Live Third-Party APIs for E2E Tests**
*   Slows down tests.
*   Introduces flakiness due to external service instability.
*   Difficult to test error scenarios (e.g., 500s from external API).

✅ **GOOD: Mock External APIs**
1.  Define API schema and examples for the third-party API in a Postman Collection.
2.  Create a Mock Server linked to this collection.
3.  Configure your application's E2E test environment to point to the Mock Server URL instead of the actual third-party API.
4.  Use the Postman API to dynamically update mock examples for specific test scenarios (e.g., simulate 400, 500 errors).

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that lead to brittle or hard-to-debug tests.

### 4.1 Asynchronous Operations in Scripts

`pm.sendRequest` is asynchronous. Always handle its promise correctly.

❌ **BAD: Ignoring `pm.sendRequest` Asynchronicity**
```javascript
// Pre-request Script
let token;
pm.sendRequest("{{BASE_URL}}/auth/token", function (err, res) {
    token = res.json().access_token; // token might not be set before next line
});
pm.environment.set("authToken", token); // This will likely be undefined
```

✅ **GOOD: Handling `pm.sendRequest` with Callbacks/Promises**
```javascript
// Pre-request Script
pm.sendRequest("{{BASE_URL}}/auth/token", function (err, res) {
    if (err) {
        console.error(err);
        return;
    }
    pm.environment.set("authToken", res.json().access_token);
});
```
*Note: For complex async flows, consider using `pm.test.async` or restructuring to avoid deep nesting.*

### 4.2 Overly Complex Scripts

Keep pre-request and test scripts focused on a single responsibility. For complex logic, consider externalizing to a shared library if using the VS Code extension, or break down into smaller, chained requests.

## 5. Error Handling and Assertions

Robust tests include explicit checks for both success and failure scenarios.

### 5.1 Comprehensive Assertions

Every request should have specific assertions (tests) to validate the response. Don't just check for `200 OK`.

❌ **BAD: Minimal Assertions**
```javascript
// Test Script
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});
```

✅ **GOOD: Detailed Assertions**
```javascript
// Test Script for POST /users
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});
pm.test("Response body has 'id' and 'name'", function () {
    const response = pm.response.json();
    pm.expect(response).to.have.property('id').that.is.a('string');
    pm.expect(response).to.have.property('name').that.equals('John Doe');
});
pm.test("Content-Type header is application/json", function () {
    pm.response.to.have.header("Content-Type", "application/json");
});
```

### 5.2 Negative Testing

Explicitly test how your API handles invalid inputs, missing authentication, or other error conditions.

```javascript
// Request: POST /users (with invalid data)
// Test Script
pm.test("Status code is 400 Bad Request", function () {
    pm.response.to.have.status(400);
});
pm.test("Error message indicates invalid input", function () {
    const response = pm.response.json();
    pm.expect(response.message).to.include("Invalid input for field 'email'");
});
```

## 6. Request/Response Patterns

Standardize how you interact with APIs.

### 6.1 API Schema Validation

Link your Postman Collections to an OpenAPI (3.x) definition in Postman's API Builder. This automatically validates requests and responses against your schema, catching inconsistencies early.

❌ **BAD: Manual Schema Checks**
*   Prone to human error.
*   Time-consuming.
*   Schema drift goes unnoticed.

✅ **GOOD: Automated API Builder Validation**
1.  Define your API using OpenAPI 3.x in Postman's API Builder.
2.  Link your Postman Collection to this API definition.
3.  Enable "Request Validation" in Postman Settings.
4.  Postman will automatically flag requests/responses that deviate from the schema.

### 6.2 Dynamic Data Generation

Use Postman's dynamic variables (`$randomInt`, `$guid`, etc.) or custom JS in pre-request scripts (e.g., `faker` library via VS Code extension) to generate unique test data.

```javascript
// Pre-request Script
// Using Postman's built-in dynamic variables
pm.environment.set("randomName", "User " + pm.variables.replaceIn("{{$randomFirstName}}"));
pm.environment.set("randomEmail", pm.variables.replaceIn("{{$randomEmail}}"));

// Request Body for POST /users
// {
//   "name": "{{randomName}}",
//   "email": "{{randomEmail}}"
// }
```

## 7. Rate Limiting

Test your API's rate limiting behavior to ensure it performs as expected under stress and handles client-side retries.

### 7.1 Testing Rate Limit Responses

Create specific requests or collection runs designed to exceed rate limits and assert the expected `429 Too Many Requests` status code and appropriate headers (e.g., `Retry-After`).

```javascript
// Request: GET /data (repeatedly in Collection Runner)
// Test Script
pm.test("Status code is 429 Too Many Requests", function () {
    pm.response.to.have.status(429);
});
pm.test("Retry-After header is present", function () {
    pm.response.to.have.header("Retry-After");
});
```

## 8. Test Organization and CI/CD Integration

Integrate Postman into your development lifecycle for continuous quality.

### 8.1 CI/CD Automation with Newman

Export Postman Collections and Environments as JSON and run them automatically in your CI/CD pipelines (GitHub Actions, Jenkins, GitLab) using Newman, Postman's command-line collection runner.

```bash
# Example CI/CD command to run a collection
newman run my-api-collection.json -e dev-environment.json --reporters cli,htmlextra
```

### 8.2 Semantic Versioning for Collections

Apply semantic versioning to your Postman Collections (e.g., `My API v1.0.0`). Store collections in source control and tag releases. This ensures traceability and stable API definitions.

### 8.3 Code Reviews for Test Scripts

Treat Postman pre-request and test scripts as first-class code. Implement code review gates for these scripts to ensure consistency, best practices, and prevent hard-coded secrets.

### 8.4 Postbot for Test Generation

Leverage Postbot (Postman's AI assistant) to suggest and generate initial test cases and autocomplete code. This accelerates test creation, but always review and refine AI-generated content to fit our specific standards.

```javascript
// Example Postbot prompt: "Generate tests for a successful user creation response."
// Postbot might suggest:
pm.test("Status code is 201", () => {
    pm.response.to.have.status(201);
});
pm.test("Response has user ID", () => {
    const responseJson = pm.response.json();
    pm.expect(responseJson.id).to.be.a('string');
});
```
*Always review and enhance Postbot's suggestions to meet our detailed assertion standards.*