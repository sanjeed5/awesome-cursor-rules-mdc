---
description: Comprehensive best practices and coding standards for utilizing Pydantic effectively in Python projects, covering code organization, performance, security, and testing.
globs: *.py
---
- **Model Definition:**
  - Use `BaseModel` to define data schemas with type annotations for clarity and automatic validation.
  - Prefer simple models that encapsulate a single concept to maintain readability and manageability.
  - Use nested models for complex data structures while ensuring each model has clear validation rules.
  - Always define a `Config` class within your model to control model behavior.

- **Validation and Error Handling:**
  - Implement built-in and custom validators to enforce data integrity.
  - Utilize `@field_validator` for field-specific rules and `@root_validator` for cross-field validation.
  - Ensure that validation errors are user-friendly and logged for debugging purposes.
  - Custom error messages should be informative and guide the user on how to correct the data.
  - Use `ValidationError` to catch and handle validation errors.

- **Performance Optimization:**
  - Consider using lazy initialization and avoid redundant validation where data is already trusted.
  - Use Pydantic's configuration options to control when validation occurs, which can significantly enhance performance in high-throughput applications.
  - Use `model_rebuild` to dynamically rebuild models when related schema change.
  - Consider using `__slots__` in the `Config` class to reduce memory footprint.

- **Code Organization and Structure:**
  - **Directory Structure:**
    - Adopt a modular structure: `src/`, `tests/`, `docs/`.
    - Models can reside in `src/models/`.
    - Validators in `src/validators/`.
    - Example:
      
      project_root/
      ├── src/
      │   ├── __init__.py
      │   ├── models/
      │   │   ├── __init__.py
      │   │   ├── user.py
      │   │   ├── item.py
      │   ├── validators/
      │   │   ├── __init__.py
      │   │   ├── user_validators.py
      │   ├── main.py  # Application entry point
      ├── tests/
      │   ├── __init__.py
      │   ├── test_user.py
      │   ├── test_item.py
      ├── docs/
      │   ├── ...
      ├── .env
      ├── pyproject.toml
      ├── README.md
      
  - **File Naming:**
    - Use snake_case for file names (e.g., `user_model.py`).
    - Name model files after the primary model they define (e.g., `user.py` for `UserModel`).
  - **Module Organization:**
    - Group related models and validators into separate modules.
    - Utilize `__init__.py` to make modules importable.
  - **Component Architecture:**
    - Employ a layered architecture (e.g., data access, business logic, presentation).
    - Pydantic models are primarily used in the data access layer.
  - **Code Splitting:**
    - Split large models into smaller, manageable components using composition.
    - Leverage inheritance judiciously.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Data Transfer Object (DTO):** Pydantic models serve as DTOs.
    - **Factory Pattern:** Create model instances using factory functions for complex initialization.
    - **Repository Pattern:** Use repositories to abstract data access and validation logic.
  - **Recommended Approaches:**
    - Centralize validation logic in dedicated validator functions.
    - Utilize nested models for complex data structures.
    - Use `BaseSettings` for managing application settings.
  - **Anti-patterns:**
    - Embedding business logic directly into models.
    - Overly complex inheritance hierarchies.
    - Ignoring validation errors.
    - Performing I/O operations within validator functions.
  - **State Management:**
    - Use immutable models whenever possible to simplify state management.
    - Consider using state management libraries like `attrs` or `dataclasses` in conjunction with Pydantic for complex applications.
  - **Error Handling:**
    - Raise `ValidationError` exceptions when validation fails.
    - Provide informative error messages.
    - Log validation errors for debugging.

- **Performance Considerations:**
  - **Optimization Techniques:**
    - Use `model_rebuild` to recompile models when their schema changes.
    - Leverage `__slots__` in the `Config` class to reduce memory footprint.
    - Use the `@cached_property` decorator to cache expensive computations.
  - **Memory Management:**
    - Be mindful of large lists or dictionaries within models, as they can consume significant memory.
    - Use generators or iterators for processing large datasets.
  - **Efficient Data Parsing:**
   -  Utilize `model_validate_json` and `model_validate` for efficient data parsing.
  - **Controlling Validation:**
    -  Use `validate_default` and `validate_assignment` options in the `Config` class to control validation occurrence.

- **Security Best Practices:**
  - **Common Vulnerabilities:**
    - Injection attacks (e.g., SQL injection) if model data is used directly in database queries.
    - Cross-site scripting (XSS) if model data is displayed in web pages without proper escaping.
    - Deserialization vulnerabilities if models are deserialized from untrusted sources.
  - **Input Validation:**
    - Always validate all incoming data using Pydantic models.
    - Use appropriate type annotations and constraints to restrict input values.
    - Sanitize input data to remove potentially harmful characters or sequences.
  - **Authentication and Authorization:**
    - Use authentication and authorization mechanisms to restrict access to sensitive data.
    - Implement role-based access control (RBAC) to grant different levels of access to different users.
  - **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure storage mechanisms for storing API keys and other secrets.
    - Mask sensitive data in logs and error messages.
  - **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement API rate limiting to prevent denial-of-service attacks.

- **Testing Approaches:**
  - **Unit Testing:**
    - Test individual models and validators in isolation.
    - Use parameterized tests to cover different input values and scenarios.
    - Verify that validation errors are raised correctly.
  - **Integration Testing:**
    - Test the interaction between models and other components of the application.
    - Use mock objects to simulate external dependencies.
  - **End-to-End Testing:**
    - Test the entire application flow from end to end.
    - Use automated testing tools to simulate user interactions.
  - **Test Organization:**
    - Organize tests into separate modules based on the component being tested.
    - Use descriptive test names to indicate the purpose of each test.
  - **Mocking and Stubbing:**
    - Use mock objects to simulate external dependencies such as databases or APIs.
    - Use stub objects to provide predefined responses for certain functions or methods.

- **Common Pitfalls and Gotchas:**
  - **Frequent Mistakes:**
    - Misusing Union Types: Using `Union` incorrectly can complicate type validation and handling.
    - Optional Fields without Default Values: Forgetting to provide a default value for optional fields can lead to `None` values causing errors in your application.
    - Incorrect Type Annotations: Assigning incorrect types to fields can cause validation to fail. For example, using `str` for a field that should be an `int`.
  - **Edge Cases:**
    - Handling complex validation logic with dependencies between fields.
    - Dealing with large or deeply nested data structures.
    - Handling different input formats (e.g., JSON, CSV).
  - **Version-Specific Issues:**
    - Be aware of breaking changes between Pydantic versions.
    - Consult the Pydantic documentation for migration guides.
  - **Compatibility Concerns:**
    - Ensure compatibility between Pydantic and other libraries used in your project.
    - Be mindful of potential conflicts with other validation libraries.

- **Tooling and Environment:**
  - **Development Tools:**
    - Use a code editor or IDE with Pydantic support (e.g., VS Code with the Pylance extension).
    - Use a static type checker like MyPy to catch type errors.
    - Use a linter like Flake8 or Pylint to enforce code style.
  - **Build Configuration:**
    - Use a build tool like Poetry or Pipenv to manage dependencies.
    - Specify Pydantic as a dependency in your project's configuration file.
  - **Linting and Formatting:**
    - Configure a linter and formatter to enforce consistent code style.
    - Use pre-commit hooks to automatically run linters and formatters before committing code.
  - **Deployment:**
    - Use a deployment platform that supports Python applications (e.g., Heroku, AWS Elastic Beanstalk, Docker).
    - Configure your deployment environment to install Pydantic and its dependencies.
  - **CI/CD:**
    - Integrate Pydantic tests into your CI/CD pipeline.
    - Automatically run tests and linters on every commit.

- **Getting Started with Pydantic:**
  - Install Pydantic with `pip install pydantic`
  - Define your data models using `BaseModel` and type hints
  - Validate your data by instantiating the data models
  - Handle validation errors using `try...except ValidationError`

- **Example:**
  python
  from pydantic import BaseModel, ValidationError
  from typing import List, Optional

  class Address(BaseModel):
    street: str
    city: str
    zip_code: Optional[str] = None

  class User(BaseModel):
    id: int
    name: str
    email: str
    addresses: List[Address]

  try:
    user_data = {
        "id": 1,
        "name": "John Doe",
        "email": "invalid-email",
        "addresses": [{
            "street": "123 Main St",
            "city": "Anytown"
        }]
    }
    user = User(**user_data)
    print(user)
  except ValidationError as e:
    print(e.json())