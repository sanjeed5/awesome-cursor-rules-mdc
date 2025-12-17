---
description: This guide provides opinionated, actionable best practices for using Insomnia IDE, focusing on modern API testing, collaboration, and CI/CD integration as of 2025.
globs: **/*
---
# Insomnia Best Practices (2025)

Insomnia is your primary tool for API interaction, debugging, and automated testing. This guide outlines the definitive best practices for leveraging Insomnia in our API-first development pipeline, emphasizing source control, AI assistance, and CI/CD integration.

## 1. Code Organization and Structure

Treat your Insomnia collections as source-controlled API contracts. They must be versioned alongside your OpenAPI specifications and application code.

### ✅ GOOD: Source-Controlled Collections with Environment Variables

Export your workspace or specific collections to JSON/YAML and commit them to Git. Use environment variables for all dynamic values, especially secrets.

```json
// my-api-collection.json (example structure)
{
  "_type": "export",
  "__export_format": 4,
  "__export_date": "2025-01-01T00:00:00.000Z",
  "__export_source": "insomnia.desktop.app:v12.0.0",
  "resources": [
    {
      "_id": "env_dev",
      "parentId": "wrk_123",
      "name": "Development",
      "data": {
        "base_url": "https://dev.api.example.com",
        "api_key": "env_API_KEY", // Reference an OS environment variable
        "bearer_token": "" // Populated by pre-request script
      },
      "color": "#63b3ed",
      "_type": "environment"
    },
    {
      "_id": "req_1",
      "parentId": "fld_abc",
      "name": "Get User Profile",
      "url": "{{ base_url }}/users/me",
      "method": "GET",
      "headers": [
        { "name": "Authorization", "value": "Bearer {{ bearer_token }}" }
      ],
      "authentication": {
        "type": "bearer",
        "token": "{{ bearer_token }}"
      },
      "_type": "request"
    }
  ]
}
```

```bash
# .gitignore
*.env.json # Ignore local environment overrides
```

### ❌ BAD: Hardcoding URLs or Secrets

Never hardcode `https://api.example.com` or `Authorization: Bearer my-secret-token` directly into requests. This breaks reproducibility and compromises security.

## 2. Common Patterns and Anti-patterns

Embrace Insomnia's AI-assisted features and CLI for efficiency and automation.

### ✅ GOOD: AI-Assisted Test Generation & Self-Healing Assertions

Leverage Insomnia's MCP client to auto-generate request bodies and validation assertions from natural language. This keeps tests in sync with evolving schemas.

```text
# Prompt in Insomnia's AI assistant for a new request
"Create a POST request to /users to register a new user with a unique email and a strong password. Assert a 201 status code and that the response contains a 'userId'."

# Prompt for assertions
"Add assertions to verify the 'Get User Profile' response has a 'name' and 'email' field, and that the email matches the one from the registration request."
```

### ❌ BAD: Manual Test Creation & Brittle Assertions

Manually crafting JSON bodies and hardcoding expected response values. This is slow, error-prone, and leads to high maintenance when APIs change. Avoid assertions that break on non-breaking schema changes.

## 3. Performance Considerations

While Insomnia Desktop is not a dedicated load testing tool, its CLI (`inso`) integrates with CI/CD for performance checks.

### ✅ GOOD: CI/CD Integration for Performance Baselines

Run collection runners via `inso` in your CI/CD pipeline to establish performance baselines and detect regressions.

```bash
# .github/workflows/api-tests.yml
name: API Tests
on: [push]
jobs:
  api-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Insomnia CLI
        run: npm install -g @getinsomnia/inso-cli
      - name: Run API Tests
        env:
          API_KEY: ${{ secrets.API_KEY }} # Pass secrets securely
        run: inso run test "My API Suite" --env "Development" --ci
```

### ❌ BAD: Ignoring Performance in API Tests

Only focusing on functional correctness. Neglecting response times or throughput checks means potential performance bottlenecks go undetected until production.

## 4. Common Pitfalls and Gotchas

Avoid common mistakes that lead to inconsistent environments or security vulnerabilities.

### ✅ GOOD: Secure Environment Management

Store sensitive environment variables (e.g., API keys, client secrets) as OS environment variables and reference them in Insomnia (`{{ env_API_KEY }}`). Never commit these to Git.

### ❌ BAD: Committing Sensitive Data

Adding `.env` files or direct secrets into your Insomnia collection JSON/YAML and committing them to your repository. This is a critical security vulnerability.

## 5. Error Handling

Comprehensive API testing includes validating error responses.

### ✅ GOOD: Asserting Expected Error Responses

Create dedicated requests and assertions to verify API behavior under error conditions (e.g., invalid input, unauthorized access, server errors).

```javascript
// Insomnia Test Script (Post-request)
const { expect } = require('chai');
const response = pm.response.json();

it('should return 400 for invalid input', function() {
  expect(pm.response.status).to.equal(400);
  expect(response.message).to.include('Invalid email format');
});
```

### ❌ BAD: Only Testing 2xx Success Paths

Assuming your API will always return a 200 OK. APIs fail, and your tests must validate that failures are handled gracefully and return expected error structures.

## 6. Request/Response Patterns

Standardize your request and response handling for consistency.

### ✅ GOOD: Dynamic Authentication with Pre-request Scripts

Use pre-request scripts to dynamically fetch and set authentication tokens (e.g., OAuth 2.0 token refresh).

```javascript
// Insomnia Pre-request Script (for a request that needs a bearer token)
const tokenRequest = await insomniac.send({
  method: 'POST',
  url: '{{ base_url }}/oauth/token',
  body: {
    grant_type: 'client_credentials',
    client_id: '{{ client_id }}',
    client_secret: '{{ client_secret }}'
  }
});

const token = tokenRequest.json.access_token;
insomniac.setEnvironmentVariable('bearer_token', token);
```

### ❌ BAD: Manual Token Copy-Pasting

Copying and pasting bearer tokens or API keys between requests. This is inefficient, error-prone, and leads to expired tokens breaking your workflow.

## 7. Rate Limiting

Explicitly test your API's rate-limiting mechanisms.

### ✅ GOOD: Dedicated Rate Limit Tests

Create a test suite that deliberately exceeds rate limits and asserts the expected `429 Too Many Requests` response, along with `Retry-After` headers.

```javascript
// Insomnia Test Script (Post-request for a rate-limited endpoint)
const { expect } = require('chai');

it('should return 429 Too Many Requests', function() {
  expect(pm.response.status).to.equal(429);
  expect(pm.response.headers.get('Retry-After')).to.exist;
});
```

### ❌ BAD: Ignoring Rate Limits

Assuming your API will never be rate-limited or that it's not a critical test case. Proper rate-limiting is essential for API stability and abuse prevention.

## 8. Test Organization

Maintain a clear, hierarchical structure for your requests and test suites.

### ✅ GOOD: Logical Folder Structure and Test Suites

Organize requests into folders reflecting your API's domain or feature areas. Group related requests into Insomnia Test Suites for logical execution.

```
My API Workspace
├── Environments
│   ├── Development
│   └── Production
├── Authentication
│   ├── POST /login
│   └── POST /refresh-token
├── Users
│   ├── GET /users/me
│   ├── POST /users
│   └── PUT /users/{id}
├── Products
│   ├── GET /products
│   └── GET /products/{id}
└── Test Suites
    ├── User Management Tests
    │   ├── Register User Test
    │   └── Get User Profile Test
    └── Product Catalog Tests
```

### ❌ BAD: Flat List of Requests

A disorganized, flat list of requests makes it impossible to navigate, understand, or run targeted tests efficiently.