---
description: This guide provides definitive, opinionated best practices for using htmx, focusing on declarative markup, secure server interaction, and efficient UI updates for modern web applications.
globs: **/*.{js,jsx}
---
# htmx Best Practices

This document outlines the definitive best practices for utilizing htmx within our projects. Adhering to these guidelines ensures maintainable, performant, and secure applications.

## 1. Code Organization and Structure

Always structure your HTML with htmx attributes as the primary driver for interactivity. Keep JavaScript minimal and reserved for complex, client-side-only interactions.

### 1.1 Modular HTML Fragments
Design your server-side templates to return small, self-contained HTML fragments. This keeps responses lean and promotes reusability.

❌ BAD: Returning a full page for a small update.
```html
<!-- Server response for a "like" action -->
<html>
  <head>...</head>
  <body>
    <header>...</header>
    <main>
      <div id="post-123">
        <!-- Entire post re-rendered -->
        <button hx-post="/like/123" hx-target="#post-123" hx-swap="outerHTML">
          Like (10)
        </button>
      </div>
    </main>
    <footer>...</footer>
  </body>
</html>
```

✅ GOOD: Returning only the updated component.
```html
<!-- Server response for a "like" action -->
<button hx-post="/like/123" hx-target="#like-button-123" hx-swap="outerHTML">
  Unlike (11)
</button>
```
```html
<!-- Initial HTML -->
<div id="post-123">
  <button id="like-button-123" hx-post="/like/123" hx-target="#like-button-123" hx-swap="outerHTML">
    Like (10)
  </button>
</div>
```

### 1.2 Attribute Placement and Inheritance
Place `hx-*` attributes on the element that triggers the action or on a parent element for inheritance. This improves readability and reduces repetition.

❌ BAD: Repeating attributes unnecessarily.
```html
<div class="item">
  <button hx-post="/delete/1" hx-target="#item-1" hx-swap="outerHTML">Delete</button>
</div>
<div class="item">
  <button hx-post="/delete/2" hx-target="#item-2" hx-swap="outerHTML">Delete</button>
</div>
```

✅ GOOD: Leveraging attribute inheritance.
```html
<div hx-target="closest .item" hx-swap="outerHTML">
  <div id="item-1" class="item">
    <button hx-post="/delete/1">Delete</button>
  </div>
  <div id="item-2" class="item">
    <button hx-post="/delete/2">Delete</button>
  </div>
</div>
```

## 2. Common Patterns and Anti-patterns

### 2.1 Declarative Data Fetching
Always use `hx-get`, `hx-post`, `hx-put`, `hx-patch`, `hx-delete` for server interactions. Avoid manual `fetch` or `XMLHttpRequest` calls for htmx-driven behavior.

❌ BAD: Mixing htmx with imperative JS for basic data fetching.
```html
<button onclick="fetch('/data').then(res => res.text()).then(html => document.getElementById('content').innerHTML = html)">Load Data</button>
```

✅ GOOD: Pure htmx for server interaction.
```html
<button hx-get="/data" hx-target="#content" hx-swap="innerHTML">Load Data</button>
```

### 2.2 Explicit Targeting and Swapping
Always define `hx-target` and `hx-swap` to control precisely where and how content updates. Default behavior (`outerHTML` on the triggering element) is often insufficient or unclear.

❌ BAD: Relying on default swap behavior for complex updates.
```html
<!-- Button is replaced, but maybe we only wanted to update a counter inside it -->
<button hx-post="/increment-counter">Click to Increment (0)</button>
```

✅ GOOD: Precise targeting and swapping.
```html
<button hx-post="/increment-counter" hx-target="this" hx-swap="innerHTML">Click to Increment (<span id="counter">0</span>)</button>
```

### 2.3 Smart Event Triggers
Use `hx-trigger` modifiers (`changed`, `delay`, `throttle`, `once`, `from:`) to optimize requests and user experience.

❌ BAD: Firing requests on every keypress for search.
```html
<input type="search" hx-get="/search" hx-target="#search-results" name="q">
```

✅ GOOD: Debouncing search input.
```html
<input type="search" hx-get="/search" hx-target="#search-results" name="q" hx-trigger="keyup changed delay:300ms">
```

### 2.4 Form Submissions
Submit forms using `hx-post` (or other HTTP verbs) directly on the form or submit button. This leverages htmx's automatic serialization.

❌ BAD: Using JavaScript to serialize and submit forms.
```html
<form id="myForm">
  <input name="name">
  <button type="button" onclick="submitForm()">Submit</button>
</form>
<script>
  function submitForm() {
    const formData = new FormData(document.getElementById('myForm'));
    fetch('/submit', { method: 'POST', body: formData });
  }
</script>
```

✅ GOOD: Declarative form submission.
```html
<form hx-post="/submit" hx-target="#form-messages" hx-swap="innerHTML">
  <input name="name">
  <button type="submit">Submit</button>
  <div id="form-messages"></div>
</form>
```

### 2.5 Loading Indicators
Always provide visual feedback for ongoing AJAX requests using the `htmx-indicator` class and `hx-indicator` attribute.

❌ BAD: No visual feedback during requests.
```html
<button hx-post="/long-operation">Process</button>
```

✅ GOOD: Clear loading state.
```html
<style>
  .htmx-indicator { opacity: 0; transition: opacity 200ms ease-in; }
  .htmx-request .htmx-indicator { opacity: 1; }
  .htmx-request.htmx-indicator { opacity: 1; }
</style>
<button hx-post="/long-operation" hx-indicator="#spinner">Process</button>
<img id="spinner" class="htmx-indicator" src="/spinner.gif" alt="Loading...">
```

### 2.6 URL Management
Use `hx-push-url` or `hx-replace-url` to update the browser's history and URL without a full page reload, maintaining user context.

❌ BAD: Full page reload for navigation.
```html
<a href="/dashboard">Dashboard</a>
```

✅ GOOD: AJAX navigation with URL update.
```html
<a hx-get="/dashboard" hx-target="#main-content" hx-push-url="/dashboard">Dashboard</a>
```

### 2.7 Extensions
Leverage htmx extensions (`hx-ext`) for common advanced functionalities (e.g., `json-enc`, `class-tools`, `morphdom`). Avoid reimplementing these features with custom JavaScript.

❌ BAD: Manually serializing JSON or complex class toggling.
```html
<form onsubmit="event.preventDefault(); sendJson(this);">...</form>
<button onclick="toggleClasses()">...</button>
```

✅ GOOD: Using htmx extensions.
```html
<form hx-post="/api/data" hx-ext="json-enc">...</form>
<div hx-ext="class-tools">
  <button classes="toggle:active on #target-div">Toggle</button>
  <div id="target-div">...</div>
</div>
```

## 3. Performance Considerations

### 3.1 Minimal HTML Responses
Ensure server responses contain only the necessary HTML fragment. Avoid sending redundant markup, CSS, or JavaScript.

### 3.2 Efficient Swapping Strategies
Choose the most efficient `hx-swap` strategy. `outerHTML` or `innerHTML` are generally preferred. Use `morph` (via `morphdom` extension) for complex, stateful updates.

### 3.3 Server-Side Caching
Implement robust server-side caching for frequently requested htmx endpoints that return static or semi-static content.

## 4. Common Pitfalls and Gotchas

### 4.1 Security: No State-Changing GET Requests
Never use `hx-get` for actions that modify server-side state (e.g., deleting a resource). Always use `hx-post`, `hx-put`, `hx-patch`, or `hx-delete` for such operations.

❌ BAD: Deleting with a GET request.
```html
<button hx-get="/delete-item/123" hx-confirm="Are you sure?">Delete Item</button>
```

✅ GOOD: Deleting with a DELETE request.
```html
<button hx-delete="/delete-item/123" hx-confirm="Are you sure?" hx-target="closest .item" hx-swap="outerHTML">Delete Item</button>
```

### 4.2 Server-Side Input Validation
Always validate all user input on the server-side. Return appropriate HTTP status codes (e.g., `400 Bad Request`) and HTML fragments with error messages.

❌ BAD: Client-side only validation.
```html
<form hx-post="/register">
  <input name="email" type="email" required>
  <button type="submit">Register</button>
</form>
<!-- Relies solely on browser validation -->
```

✅ GOOD: Server-side validation with htmx feedback.
```html
<form hx-post="/register" hx-target="#form-errors" hx-swap="innerHTML">
  <input name="email" type="email" required>
  <button type="submit">Register</button>
  <div id="form-errors">
    <!-- Server returns validation errors here -->
  </div>
</form>
<!-- Server response for invalid email: <div id="form-errors"><p class="error">Invalid email format.</p></div> -->
```

### 4.3 XSS Prevention
Ensure all user-generated content rendered by the server is properly HTML-escaped. Use auto-escaping template engines (e.g., Jinja2, Go's `html/template`).

## 5. Testing Approaches

### 5.1 Server-Side Integration Tests
Focus your testing efforts on the server-side. Write integration tests for your endpoints that verify the correctness of the HTML fragments returned by htmx requests.

### 5.2 End-to-End (E2E) Testing
Use E2E testing frameworks (e.g., Playwright, Cypress) to simulate user interactions and verify the overall application flow and UI updates. This is crucial for htmx applications as much of the logic lives in the HTML.

### 5.3 Avoid Client-Side Unit Tests for htmx Logic
Since htmx logic is declarative in HTML, writing traditional client-side JavaScript unit tests for htmx attributes is an anti-pattern. The framework handles the client-side mechanics; your tests should focus on the server's responses and the resulting DOM state.