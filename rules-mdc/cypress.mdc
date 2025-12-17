---
description: This guide defines definitive best practices for writing maintainable, performant, and reliable Cypress end-to-end tests, leveraging modern patterns and avoiding common pitfalls.
globs: **/*.{js,ts,cy.js,cy.ts}
---
# Cypress Best Practices

Cypress is our go-to for robust E2E testing. Follow these rules to ensure our test suite remains fast, stable, and easy to maintain.

## 1. Code Organization & Structure

Adhere to the standard Cypress folder structure. This promotes consistency and discoverability.

**Rule:** Keep a single `cypress.config.ts` at the root.
**Rule:** Place E2E tests in `cypress/e2e`.
**Rule:** Store reusable commands in `cypress/support/commands.ts`.
**Rule:** Use `cypress/fixtures` for static test data.

```typescript
// cypress.config.ts
import { defineConfig } from 'cypress';

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000', // Always set a global base URL
    retries: {
      runMode: 2, // Allow retries in CI for flaky tests
      openMode: 0,
    },
    setupNodeEvents(on, config) {
      // Implement cy.task() here for server-side operations
    },
  },
});
```

## 2. Test Isolation

Each test (`it` block) must be independent and not rely on the state of previous tests. This enables parallelization and makes debugging straightforward.

❌ **BAD:** Dependent tests
```typescript
// test-suite.cy.ts
describe('User Flow', () => {
  it('should register a user', () => { /* ... */ }); // Creates user
  it('should log in the registered user', () => { /* ... */ }); // Depends on previous test
});
```

✅ **GOOD:** Isolated tests using `beforeEach`
```typescript
// test-suite.cy.ts
describe('User Management', () => {
  beforeEach(() => {
    // Programmatically create user and log in via API or custom command
    cy.apiLogin('testuser@example.com', 'password123');
    cy.visit('/dashboard');
  });

  it('should display user dashboard', () => {
    cy.getBySel('dashboard-header').should('contain', 'Welcome, Test User');
  });

  it('should allow user to update profile', () => {
    cy.getBySel('profile-link').click();
    cy.getBySel('name-input').type('New Name');
    cy.getBySel('save-button').click();
    cy.getBySel('success-message').should('be.visible');
  });
});
```

## 3. Element Selection

Prioritize stable, test-specific attributes. This decouples selectors from styling or content changes.

**Rule:** Always use `data-cy` attributes for element selection.
**Rule:** Use `cy.contains()` only when the displayed text is critical to the test's purpose.

❌ **BAD:** Brittle selectors
```typescript
cy.get('.btn-primary').click(); // Coupled to styling
cy.get('#submitButton').click(); // Coupled to JS/CSS ID
cy.get('div:nth-child(2) > p').should('be.visible'); // Highly brittle DOM structure
```

✅ **GOOD:** Resilient selectors with `data-cy`
```typescript
// In cypress/support/commands.ts
Cypress.Commands.add('getBySel', (selector: string, ...args: any[]) => {
  return cy.get(`[data-cy="${selector}"]`, ...args);
});

// In your test file
cy.getBySel('submit-button').click(); // Best practice
cy.getBySel('user-profile-name').should('contain', 'John Doe');
cy.contains('Submit Order').click(); // Only if "Submit Order" text is critical
```

## 4. Mocking Strategies

Control external dependencies to ensure fast, reliable, and deterministic tests.

**Rule:** Use `cy.intercept()` for all network requests (XHR/Fetch).
**Rule:** Use `cy.fixture()` to provide static response data.
**Rule:** Use `cy.stub()` for client-side function mocking.
**Rule:** For server-side setup/teardown, use `cy.task()` to execute Node.js code.

❌ **BAD:** Relying on live API responses
```typescript
it('should load products', () => {
  cy.visit('/products');
  cy.get('.product-card').should('have.length.gt', 0); // Fails if API is down or slow
});
```

✅ **GOOD:** Intercepting network requests with fixtures
```typescript
it('should load products from fixture', () => {
  cy.intercept('GET', '/api/products', { fixture: 'products.json' }).as('getProducts');
  cy.visit('/products');
  cy.wait('@getProducts');
  cy.getBySel('product-card').should('have.length', 3);
});
```

## 5. Avoiding Arbitrary Waits

Cypress automatically retries assertions. Avoid `cy.wait()` unless explicitly waiting for a network request to complete.

❌ **BAD:** Arbitrary waits
```typescript
cy.get('button').click();
cy.wait(2000); // Unnecessary, slows down tests, can cause flakiness
cy.get('.success-message').should('be.visible');
```

✅ **GOOD:** Leveraging Cypress's retry-ability
```typescript
cy.getBySel('submit-button').click();
cy.getBySel('success-message').should('be.visible'); // Cypress retries until visible
```
✅ **GOOD:** Waiting for specific network requests
```typescript
cy.intercept('POST', '/api/users', { statusCode: 201 }).as('createUser');
cy.getBySel('register-form').submit();
cy.wait('@createUser').its('response.statusCode').should('eq', 201);
```

## 6. Performance & Real-world Scenarios

Optimize test execution and simulate real user interactions effectively.

**Rule:** Log in programmatically via `cy.request()` or a custom command in `beforeEach` instead of using the UI. This is significantly faster.
**Rule:** Use `cy.session()` for persistent login state across tests within a spec file, reducing redundant login calls.
**Rule:** Leverage `cy.prompt()` (available 2025) for natural language test generation and self-healing selectors.

```typescript
// cypress/support/commands.ts
Cypress.Commands.add('apiLogin', (email, password) => {
  cy.session([email, password], () => { // Cache session for faster subsequent runs
    cy.request('POST', '/api/login', { email, password })
      .its('body.token')
      .then((token) => {
        localStorage.setItem('jwt', token); // Store token for authenticated requests
      });
  }, {
    cacheAcrossSpecs: true // Cache across spec files for even faster runs
  });
});

// In your test file
describe('Authenticated Features', () => {
  beforeEach(() => {
    cy.apiLogin('user@example.com', 'password123');
    cy.visit('/dashboard');
  });
  // ... tests ...
});
```

## 7. Test Naming & Readability

Write clear, descriptive test names that explain the intent and expected outcome.

❌ **BAD:** Vague test names
```typescript
it('test 1', () => { /* ... */ });
it('login works', () => { /* ... */ });
```

✅ **GOOD:** Descriptive test names
```typescript
describe('Authentication', () => {
  it('should allow a valid user to log in and redirect to dashboard', () => { /* ... */ });
  context('when credentials are invalid', () => {
    it('should display an error message for incorrect password', () => { /* ... */ });
  });
});
```

## 8. Avoiding External Sites

Keep E2E tests focused on our application. If an external origin is unavoidable, use `cy.origin()`.

❌ **BAD:** Automating external sites directly
```typescript
cy.visit('https://our-app.com');
cy.get('.external-link').click();
cy.url().should('include', 'external-provider.com');
cy.get('#external-form').type('data'); // Unreliable, out of our control
```

✅ **GOOD:** Using `cy.origin()` for controlled external interactions
```typescript
it('should handle OAuth redirect', () => {
  cy.visit('/login');
  cy.getBySel('oauth-login-button').click();

  cy.origin('https://external-oauth.com', () => {
    cy.get('#username').type('oauth_user');
    cy.get('#password').type('oauth_pass');
    cy.get('#submit').click();
  });

  cy.url().should('include', '/dashboard'); // Back to our app
});
```