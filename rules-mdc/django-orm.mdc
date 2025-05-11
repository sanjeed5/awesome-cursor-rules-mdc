---
description: This rule file provides comprehensive best practices for Django's Object-Relational Mapper (ORM), covering code organization, performance, security, testing, and common pitfalls. It aims to guide developers in building efficient, maintainable, and secure Django applications.
globs: *.py
---
## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

- **Application-Centric Structure:** Organize your Django project around applications (apps), where each app represents a specific feature or module.
- **Dedicated 'models' Module:** Keep all model definitions within the `models.py` file of each app.
- **Separate Queries:** For complex queries, consider creating a `queries.py` or `managers.py` file within your app to house custom querysets and managers.
- **Consistent Naming:** Use consistent naming conventions for app and model names.


myproject/
 ├── myproject/
 │   ├── __init__.py
 │   ├── asgi.py
 │   ├── settings.py
 │   ├── urls.py
 │   ├── wsgi.py
 ├── app1/
 │   ├── __init__.py
 │   ├── admin.py
 │   ├── apps.py
 │   ├── models.py
 │   ├── managers.py  # Custom managers and querysets
 │   ├── views.py
 │   ├── urls.py
 │   ├── forms.py
 │   ├── tests.py
 │   └── migrations/
 ├── app2/
 │   └── ...
 └── manage.py


### 1.2. File Naming Conventions

- **Models:** Use singular nouns and CamelCase (e.g., `UserProfile`, `BlogPost`).
- **Fields:** Use lowercase with underscores (snake_case, e.g., `first_name`, `date_joined`).
- **Managers/Querysets:** Use descriptive names related to their functionality (e.g., `ActiveUsersManager`, `PublishedPostsQuerySet`).
- **Files:** Use lowercase with underscores (snake_case, e.g., `models.py`, `forms.py`, `urls.py`).

### 1.3. Module Organization

- **Keep Models Concise:** Limit the number of models in each app to a manageable amount (ideally under 10). If an app becomes too large, refactor it into smaller, more focused apps.
- **Avoid Circular Imports:** Be mindful of circular dependencies between models and other modules within the same app or across apps.  Use string references in `ForeignKey` and `ManyToManyField` definitions to avoid immediate import requirements.
- **Utilize Abstract Base Classes:** For common fields across multiple models, define an abstract base class and inherit from it. This promotes code reuse and reduces redundancy. Don't forget to set `abstract = True` in the `Meta` class.
- **Consider Proxy Models:** Use proxy models (`proxy = True` in `Meta`) to change a model's behavior without altering the database schema. This is useful for adding methods or changing the default ordering.

### 1.4. Component Architecture

- **DRY (Don't Repeat Yourself):** Create reusable components such as custom model fields, managers, and querysets.  
- **Model Mixins:** Use model mixins to add common functionality to multiple models (e.g., a mixin for soft deletion or timestamping).
- **Form Abstraction:** Create reusable form classes and widgets for data input and validation.
- **Template Tags and Filters:** Develop custom template tags and filters to simplify common tasks in templates related to model data.

### 1.5. Code Splitting Strategies

- **App-Based Splitting:** Organize models and related code within distinct Django apps.
- **Manager-Based Splitting:** Divide complex queries into custom managers and querysets.
- **Mixin-Based Splitting:** Utilize mixins to compartmentalize functionality across models.
- **Signal-Based Splitting:** Implement signals for decoupling operations, such as triggering tasks when a model is saved or deleted.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

- **Repository Pattern:** Abstract data access logic behind a repository interface, allowing you to switch between different data sources without modifying the rest of your application.
- **Unit of Work Pattern:** Group multiple database operations into a single transaction to ensure atomicity and consistency.
- **Specification Pattern:** Encapsulate query logic into reusable specification objects, allowing you to combine and reuse queries across different parts of your application.
- **Model Decorators:** Use decorators to add functionality to models without modifying the original model class. Useful for things like caching calculated properties.

### 2.2. Recommended Approaches for Common Tasks

- **Filtering Data:** Use `filter()` and `exclude()` for efficient data retrieval.
- **Ordering Data:** Utilize `order_by()` to specify the order of results.
- **Aggregating Data:** Employ `aggregate()` to calculate summary statistics.
- **Creating and Updating Objects:** Use `create()` and `save()` for object manipulation.
- **Deleting Objects:** Use `delete()` for removing objects.
- **Prefetch Related:** Use `select_related` for ForeignKey and OneToOneField relationships, and `prefetch_related` for ManyToManyField and reverse ForeignKey relationships.

### 2.3. Anti-patterns and Code Smells

- **Overusing `raw()` Queries:** Prefer Django ORM methods over raw SQL queries to maintain database abstraction and security.
- **Ignoring Database Indexes:** Neglecting to add indexes to frequently queried columns can lead to significant performance bottlenecks. Use `db_index = True` and `Index` in Meta to specify indices.
- **Using `null=True` for String-Based Fields:** Avoid `null=True` for `CharField` and `TextField` fields. Use `blank=True` and an empty string (`''`) instead.
- **Creating Large Querysets without Pagination:** Avoid retrieving large datasets without pagination. Use Django's `Paginator` to break results into manageable chunks.
- **Performing Logic in Templates:** Keep templates focused on presentation. Move complex logic to views or custom template tags.
- **N+1 Query Problem:** Accessing related objects in a loop can cause the N+1 query problem. Use `select_related()` and `prefetch_related()` to mitigate this.
- **Inefficient Use of Exists/Count:** Prefer `exists()` over `count()` when simply checking for the existence of objects.
- **Unnecessary Database Hits:** Optimize data access patterns to minimize the number of database queries.

### 2.4. State Management

- **Database as Primary State:** Rely on the database as the single source of truth for application state.
- **Session Management:** Utilize Django's session framework for storing user-specific state.
- **Caching:** Implement caching strategies to reduce database load and improve response times.
- **Avoid Global Mutable State:**  Minimize the use of global mutable state, as it can lead to unexpected behavior and concurrency issues.

### 2.5. Error Handling

- **Catch Database Exceptions:** Handle database exceptions (e.g., `IntegrityError`, `OperationalError`) gracefully.
- **Use Transactions:** Wrap multiple database operations in a transaction to ensure atomicity.
- **Log Errors:** Log errors and exceptions for debugging and monitoring purposes.  Use `logger.exception()` to capture the traceback.
- **Custom Exception Handling Middleware:** Consider writing custom middleware to catch and handle database errors at a global level, providing user-friendly error messages.
- **Validate Data Before Saving:** Always validate data before saving it to the database to prevent errors and data inconsistencies. Use Django's built-in validators or create custom validators.

## 3. Performance Considerations

### 3.1. Optimization Techniques

- **Query Optimization:** Use `select_related` and `prefetch_related` to reduce the number of database queries.
- **Indexing:** Add indexes to frequently queried columns.
- **Caching:** Implement caching strategies for frequently accessed data.
- **Pagination:** Use pagination for large datasets.
- **Batch Operations:** Use `bulk_create`, `bulk_update` and `bulk_delete` for bulk operations.
- **Defer/Only:** use `defer` to exclude fields from the query or `only` to only include specific fields in the query.
- **Conditional Expressions:** Use `Case`, `When`, and `Value` within queries to perform conditional logic directly in the database.

### 3.2. Memory Management

- **Limit Queryset Size:** Avoid retrieving large datasets into memory. Use pagination or generators.
- **Use Generators:** For processing large datasets, use generators to process data in chunks.
- **Garbage Collection:** Understand Python's garbage collection mechanisms and how they affect memory usage.
- **Profiling:** Use profiling tools to identify memory leaks and optimize memory usage.

### 3.3. Rendering Optimization

- **Template Caching:** Cache frequently used templates to reduce rendering time.
- **Context Processors:** Minimize the data passed to templates via context processors.
- **Lazy Loading:** Use lazy loading for images and other resources to improve initial page load time.  Consider using template tags like `{% static %}` and `{% get_static_prefix %}` effectively.

### 3.4. Bundle Size Optimization

- **Remove Unused Code:** Eliminate unused models, fields, and queries.
- **Optimize Static Files:** Minify and compress static files (CSS, JavaScript, images).
- **Code Splitting:**  Split JavaScript and CSS into smaller chunks for efficient loading.

### 3.5. Lazy Loading

- **Load Related Data on Demand:** Use lazy loading for related data to improve initial performance. Consider using Django Rest Framework serializers with `depth=0` and selectively increasing the depth when needed.
- **Database Views:** Create database views for complex queries to improve performance.
- **QuerySet Iterators:** Instead of converting large querysets into lists, iterate over them using `.iterator()` for memory efficiency.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

- **SQL Injection:** Prevent SQL injection by using Django ORM methods instead of raw SQL queries. Always parameterize your queries.
- **Cross-Site Scripting (XSS):** Sanitize user input and escape output in templates.
- **Cross-Site Request Forgery (CSRF):** Use Django's CSRF protection mechanisms.
- **Authentication and Authorization:** Implement robust authentication and authorization mechanisms.
- **Session Hijacking:** Secure sessions using HTTPS and appropriate session settings.

### 4.2. Input Validation

- **Use Django Forms:** Employ Django Forms for validating user input.
- **Sanitize User Input:** Sanitize user input to prevent XSS attacks.
- **Validate Data Types:** Validate data types to prevent database errors.
- **Limit Input Length:** Restrict input length to prevent buffer overflows.
- **Custom Validators:** Create custom validators for complex validation logic.
- **Consider using DRF Serializers:** Use Django Rest Framework Serializers to add an extra layer of validation to your models and data.

### 4.3. Authentication and Authorization

- **Use Django's Built-in Authentication:** Leverage Django's built-in authentication framework.
- **Implement Role-Based Access Control (RBAC):** Define roles and permissions for different user types.
- **Enforce Password Policies:** Enforce strong password policies.
- **Use Multi-Factor Authentication (MFA):** Implement MFA for sensitive accounts.
- **Rate Limiting:** Implement rate limiting to prevent brute-force attacks.

### 4.4. Data Protection

- **Use HTTPS:** Use HTTPS to encrypt data in transit.
- **Encrypt Sensitive Data:** Encrypt sensitive data at rest.
- **Store Passwords Securely:** Store passwords using a strong hashing algorithm (e.g., bcrypt, Argon2).
- **Regularly Back Up Data:** Regularly back up your database.
- **Audit Logging:** Implement audit logging to track user activity and data changes.

### 4.5. Secure API Communication

- **Use API Keys:** Use API keys for authenticating API requests.
- **Implement OAuth 2.0:** Implement OAuth 2.0 for secure API authorization.
- **Validate API Input:** Validate API input to prevent injection attacks.
- **Rate Limiting:** Implement rate limiting to prevent API abuse.
- **API Versioning:** Use API versioning to manage changes to your API.

## 5. Testing Approaches

### 5.1. Unit Testing

- **Test Model Methods:** Unit test model methods to ensure they behave as expected.
- **Test Custom Managers:** Unit test custom managers to verify their query logic.
- **Test Validators:** Unit test validators to ensure they correctly validate data.
- **Mock Database Dependencies:** Use mocking to isolate model tests from the database.

### 5.2. Integration Testing

- **Test Views and Models:** Integrate test views and models to ensure they work together correctly.
- **Test Forms and Models:** Integrate test forms and models to verify data validation and persistence.
- **Use Test Fixtures:** Use test fixtures to create consistent test data.
- **Test Database Interactions:** Test database interactions to ensure they are efficient and correct.

### 5.3. End-to-End Testing

- **Test User Workflows:** End-to-end test user workflows to ensure they are functional and user-friendly.
- **Test API Endpoints:** End-to-end test API endpoints to ensure they are secure and functional.
- **Use Selenium or Similar Tools:** Use Selenium or similar tools for automating end-to-end tests.
- **Test Across Different Browsers:** Test across different browsers to ensure compatibility.

### 5.4. Test Organization

- **Organize Tests by App:** Organize tests by app to improve maintainability.
- **Use Descriptive Test Names:** Use descriptive test names to clearly indicate what is being tested.
- **Follow a Consistent Test Structure:** Follow a consistent test structure to improve readability.
- **Use Test Runners:** Use test runners to automate test execution.

### 5.5. Mocking and Stubbing

- **Mock Database Queries:** Mock database queries to isolate tests from the database.
- **Stub External Dependencies:** Stub external dependencies to control test environments.
- **Use Django's `override_settings` Decorator:** Use Django's `override_settings` decorator to modify settings during tests.
- **Use `patch` for Mocking:** Use `unittest.mock.patch` for mocking objects and functions.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

- **Not Using Indexes:** Forgetting to add indexes to frequently queried columns.
- **Overusing `null=True`:** Using `null=True` for string-based fields.
- **Ignoring Database Transactions:** Not using database transactions for critical operations.
- **Creating Large Querysets:** Creating large querysets without pagination.
- **Performing Logic in Templates:** Performing logic in templates instead of views.
- **N+1 Query Problem:** Not addressing the N+1 query problem.

### 6.2. Edge Cases

- **Unicode Handling:** Handling Unicode characters correctly.
- **Time Zones:** Dealing with time zones accurately.
- **Concurrency Issues:** Managing concurrency issues in multi-threaded environments.
- **Database-Specific Differences:** Handling database-specific differences.

### 6.3. Version-Specific Issues

- **Deprecated Features:** Being aware of deprecated features and migrating to newer alternatives.
- **Compatibility Changes:** Handling compatibility changes between Django versions.

### 6.4. Compatibility Concerns

- **Python Version:** Ensuring compatibility with the supported Python versions.
- **Database Compatibility:** Ensuring compatibility with the chosen database.
- **Third-Party Libraries:** Ensuring compatibility with third-party libraries.

### 6.5. Debugging

- **Use Django's Debug Toolbar:** Use Django's Debug Toolbar for profiling and debugging.
- **Inspect Database Queries:** Inspect database queries to identify performance bottlenecks.
- **Log Errors and Exceptions:** Log errors and exceptions for debugging purposes.
- **Use a Debugger:** Use a debugger to step through code and inspect variables.
- **Read Stack Traces Carefully:** Read stack traces carefully to understand the source of errors.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

- **IDE:** PyCharm, VS Code with Python extension.
- **Database Client:** pgAdmin, MySQL Workbench, DB Browser for SQLite.
- **Virtual Environment Manager:** virtualenv, pipenv, conda.
- **Debug Toolbar:** Django Debug Toolbar.
- **Profiling Tools:** cProfile, line_profiler.

### 7.2. Build Configuration

- **Use a Virtual Environment:** Use a virtual environment to isolate project dependencies.
- **Specify Dependencies:** Specify dependencies in a `requirements.txt` or `Pipfile`.
- **Use a Build System:** Use a build system (e.g., Make, Fabric) to automate build tasks.
- **Configure Static File Handling:** Configure static file handling correctly for production.

### 7.3. Linting and Formatting

- **Use a Linter:** Use a linter (e.g., pylint, flake8) to enforce code style.
- **Use a Formatter:** Use a formatter (e.g., black, autopep8) to automatically format code.
- **Configure Editor Integration:** Configure editor integration to automatically lint and format code on save.
- **Follow PEP 8:** Adhere to PEP 8 style guidelines.

### 7.4. Deployment

- **Use a Production-Ready Web Server:** Use a production-ready web server (e.g., Gunicorn, uWSGI).
- **Use a Reverse Proxy:** Use a reverse proxy (e.g., Nginx, Apache) to handle static files and load balancing.
- **Use a Database Server:** Use a dedicated database server (e.g., PostgreSQL, MySQL).
- **Configure Caching:** Configure caching to improve performance.
- **Secure Deployment:** Secure the deployment environment with appropriate firewalls and security settings.
- **Use Environment Variables:** Store sensitive configuration data (e.g., database passwords, API keys) in environment variables.

### 7.5. CI/CD Integration

- **Use a CI/CD System:** Use a CI/CD system (e.g., Jenkins, Travis CI, GitLab CI) to automate testing and deployment.
- **Run Tests Automatically:** Run tests automatically on every commit.
- **Deploy Automatically:** Deploy automatically on successful builds.
- **Use Infrastructure as Code:** Use infrastructure as code (e.g., Terraform, Ansible) to automate infrastructure provisioning.

---

By consistently applying these best practices, you'll establish a robust foundation for building high-quality Django applications that are performant, maintainable, and secure.