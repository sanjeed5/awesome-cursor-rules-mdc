---
description: Comprehensive guide to Django best practices covering code organization, performance, security, testing, and common pitfalls. This rule ensures adherence to community standards for maintainable and efficient Django applications.
globs: *.py
---
# Django Best Practices and Coding Standards

This guide outlines best practices for developing Django applications to ensure code quality, maintainability, performance, and security.

## 1. Code Organization and Structure

- **Directory Structure:**
    - **Project Root:** Manage.py, settings, urls, wsgi.
    - **Apps:** Each app should reside in its own directory.
    - **Static Files:** Separate directory for static files (CSS, JavaScript, images).
    - **Templates:** Dedicated directory for HTML templates.
    - **Media:** For user-uploaded files.
    - **Tests:** Tests for each app in a separate tests/ directory.

- **File Naming Conventions:**
    - Use lowercase letters with underscores (snake_case) for Python files and variables (e.g., `models.py`, `user_profile`).
    - Classes should use CamelCase (e.g., `UserProfile`).
    - Function names should use snake_case (e.g., `get_user_profile`).

- **Module Organization:**
    - Each app should be a self-contained module.
    - Group related functionalities into submodules within the app.
    - Use `__init__.py` files to define packages.

- **Component Architecture:**
    - Follow the Model-View-Template (MVT) architecture.
    - Keep models lean and focused on data representation.
    - Use views for request handling and business logic.
    - Templates should focus on presentation.

- **Code Splitting Strategies:**
    - Break down large views into smaller, reusable functions or class-based views.
    - Use mixins to share functionality across multiple views.
    - Implement custom template tags and filters for reusable template logic.
    - Consider using Celery for asynchronous task processing.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Model-View-Template (MVT):** Django's core architectural pattern.
    - **Form Objects:** Use Django's forms for handling user input and validation.
    - **Signals:** Implement signals for decoupled event handling.
    - **Managers:** Use custom model managers to encapsulate query logic.
    - **DRY (Don't Repeat Yourself):**  Avoid redundant code by extracting reusable components.

- **Recommended Approaches:**
    - **Database Interactions:** Use Django's ORM for database interactions. Avoid raw SQL queries unless necessary.
    - **Form Handling:** Leverage Django's form classes for data validation and rendering forms.
    - **URL Routing:** Use named URL patterns for reverse URL lookup.
    - **Template Inheritance:** Utilize template inheritance to create a consistent look and feel.

- **Anti-patterns and Code Smells:**
    - **Fat Models:** Avoid putting too much business logic in models. Use services or utility functions instead.
    - **Complex Templates:** Keep templates simple and focused on presentation. Move complex logic to views or custom template tags.
    - **Hardcoded Values:** Avoid hardcoding values in code. Use settings or environment variables.
    - **Ignoring Security:** Neglecting security best practices can lead to vulnerabilities.

- **State Management:**
    - Use Django's session framework for managing user sessions.
    - Avoid storing sensitive data in sessions.
    - Consider using a dedicated caching system (e.g., Redis, Memcached) for storing frequently accessed data.

- **Error Handling:**
    - Implement proper error handling using `try...except` blocks.
    - Use Django's logging framework to log errors and warnings.
    - Display user-friendly error messages.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Database Optimization:** Use indexes on frequently queried fields.
    - **Caching:** Implement caching at various levels (e.g., template caching, view caching, database caching).
    - **Query Optimization:** Use `select_related` and `prefetch_related` to reduce the number of database queries.
    - **Gzip Compression:** Enable Gzip compression for static files and responses.

- **Memory Management:**
    - Avoid loading large datasets into memory. Use iterators or generators.
    - Close database connections and file handles properly.
    - Be mindful of memory leaks.

- **Rendering Optimization:**
    - Minimize the use of complex template logic.
    - Use template caching to cache rendered templates.
    - Optimize images and other static assets.

- **Lazy Loading Strategies:**
    - Implement lazy loading for images and other resources that are not immediately visible.
    - Use pagination to display large datasets in smaller chunks.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Cross-Site Scripting (XSS):** Prevent XSS by escaping user input in templates and using Django's `escape` filter.
    - **Cross-Site Request Forgery (CSRF):** Protect against CSRF attacks by using Django's CSRF protection.
    - **SQL Injection:** Use Django's ORM to prevent SQL injection attacks. Avoid raw SQL queries.
    - **Authentication and Authorization Issues:** Implement proper authentication and authorization mechanisms.

- **Input Validation:**
    - Validate all user input on both the client-side and server-side.
    - Use Django's form validation to validate data.
    - Sanitize user input to remove potentially harmful characters.

- **Authentication and Authorization:**
    - Use Django's built-in authentication system for user authentication.
    - Implement proper authorization checks to restrict access to sensitive resources.
    - Use Django's permission system to manage user permissions.

- **Data Protection:**
    - Store sensitive data securely using encryption.
    - Use HTTPS to encrypt communication between the client and server.
    - Protect against data breaches by implementing proper access controls and monitoring.

- **Secure API Communication:**
    - Use authentication tokens or API keys to authenticate API requests.
    - Implement rate limiting to prevent abuse.
    - Validate all API input.

## 5. Testing Approaches

- **Unit Testing:**
    - Write unit tests for individual functions and classes.
    - Use Django's testing framework to write and run tests.
    - Mock external dependencies to isolate units of code.

- **Integration Testing:**
    - Write integration tests to test the interaction between different components.
    - Test database interactions, form handling, and view rendering.

- **End-to-End Testing:**
    - Write end-to-end tests to test the entire application flow.
    - Use tools like Selenium or Cypress to automate browser testing.

- **Test Organization:**
    - Organize tests into separate test suites for each app or component.
    - Use meaningful test names to describe what each test does.
    - Keep tests concise and focused.

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate units of code during testing.
    - Avoid mocking external dependencies unless necessary.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - **Incorrect Database Configuration:** Ensure that the database settings are configured correctly.
    - **Improper Static File Handling:** Configure static file serving properly in production.
    - **Not Using Django's Built-in Features:** Leverage Django's built-in features instead of reinventing the wheel.
    - **Ignoring Security Warnings:** Address any security warnings or vulnerabilities reported by Django's security checks.

- **Edge Cases:**
    - **Handling Time Zones:** Be mindful of time zones and use Django's time zone support.
    - **Dealing with Unicode:** Handle Unicode characters correctly.
    - **Managing File Uploads:** Implement proper file upload handling to prevent security vulnerabilities.

- **Version-Specific Issues:**
    - Be aware of compatibility issues between different Django versions.
    - Consult the Django release notes for information about breaking changes.

- **Compatibility Concerns:**
    - Ensure that your code is compatible with different browsers and devices.
    - Test your application on different platforms.

- **Debugging Strategies:**
    - Use Django's debugging tools to identify and fix errors.
    - Use logging to track the execution flow of your application.
    - Consult the Django documentation and community forums for help.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **IDE:** PyCharm, VS Code, Sublime Text
    - **Package Manager:** pip, uv
    - **Database Client:** pgAdmin, MySQL Workbench

- **Build Configuration:**
    - Use a `requirements.txt` file to manage dependencies.
    - Use a virtual environment to isolate project dependencies.

- **Linting and Formatting:**
    - Use Black to format code automatically.
    - Use flake8 to lint code and identify potential errors.

- **Deployment Best Practices:**
    - Use a production-ready web server (e.g., Gunicorn, uWSGI).
    - Configure a reverse proxy (e.g., Nginx, Apache).
    - Use a database server (e.g., PostgreSQL, MySQL).

- **CI/CD Integration:**
    - Integrate your project with a CI/CD pipeline to automate testing and deployment.
    - Use tools like Jenkins, Travis CI, or GitLab CI.

## Additional Notes

- Always follow PEP 8 guidelines for Python code.
- Write clear and concise comments.
- Keep your code DRY (Don't Repeat Yourself).
- Regularly update your dependencies.
- Monitor your application for performance and security issues.