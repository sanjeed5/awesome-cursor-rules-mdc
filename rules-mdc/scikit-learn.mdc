---
description: Enforces best practices and coding standards for scikit-learn projects, promoting maintainability, performance, and security. This rule provides guidelines on code organization, common patterns, performance optimization, testing, and common pitfalls.
globs: *.py
---
# Scikit-learn Best Practices and Coding Standards

This document outlines best practices and coding standards for developing with scikit-learn, aiming to improve code quality, maintainability, performance, and security.

## 1. Code Organization and Structure

- **Directory Structure:**
    - Organize your project into logical modules or components. A typical structure might include:
        - `src/`: Source code for your project.
            - `models/`: Contains the implementation of machine learning models.
            - `features/`: Feature engineering and processing logic.
            - `utils/`: Utility functions and helper modules.
        - `data/`: Datasets used for training and testing.
            - `raw/`: Original, unprocessed data.
            - `processed/`: Cleaned and preprocessed data.
        - `notebooks/`: Jupyter notebooks for experimentation and analysis.
        - `tests/`: Unit and integration tests.
        - `scripts/`: Scripts for training, evaluation, and deployment.
        - `config/`: Configuration files (e.g., YAML, JSON).
    - Example:

      
      project_root/
      ├── src/
      │   ├── models/
      │   │   ├── __init__.py
      │   │   ├── model_1.py
      │   │   └── model_2.py
      │   ├── features/
      │   │   ├── __init__.py
      │   │   ├── feature_engineering.py
      │   │   └── feature_selection.py
      │   ├── utils/
      │   │   ├── __init__.py
      │   │   └── helpers.py
      │   ├── __init__.py
      │   └── main.py
      ├── data/
      │   ├── raw/
      │   │   └── data.csv
      │   └── processed/
      │       └── processed_data.csv
      ├── notebooks/
      │   └── exploratory_data_analysis.ipynb
      ├── tests/
      │   ├── __init__.py
      │   ├── test_models.py
      │   └── test_features.py
      ├── scripts/
      │   ├── train.py
      │   └── evaluate.py
      ├── config/
      │   └── config.yaml
      ├── README.md
      └── requirements.txt
      

- **File Naming Conventions:**
    - Use descriptive and consistent names for files and modules.
    - Prefer snake_case for Python files and variables (e.g., `model_training.py`, `n_samples`).

- **Module Organization:**
    - Each module should have a clear and specific purpose.
    - Use `__init__.py` files to define packages and modules within directories.
    - Minimize dependencies between modules to improve maintainability.

- **Component Architecture:**
    - Design your application with loosely coupled components.
    - Implement interfaces or abstract classes to define contracts between components.
    - Use dependency injection to manage dependencies and promote testability.

- **Code Splitting Strategies:**
    - Split large files into smaller, more manageable modules.
    - Group related functions and classes into modules based on functionality.
    - Consider splitting code based on layers (e.g., data access, business logic, presentation).

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Pipeline:** Use scikit-learn's `Pipeline` class to chain together multiple data preprocessing and modeling steps. This helps prevent data leakage and ensures consistent data transformations.
    - **Model Selection:** Employ techniques like cross-validation and grid search to select the best model and hyperparameters for your data. Use `GridSearchCV` or `RandomizedSearchCV`.
    - **Ensemble Methods:** Leverage ensemble methods like Random Forests, Gradient Boosting, and Voting Classifiers to improve model accuracy and robustness.
    - **Custom Transformers:** Create custom transformers using `BaseEstimator` and `TransformerMixin` to encapsulate complex feature engineering logic.

- **Recommended Approaches:**
    - **Data Preprocessing:** Always split your data into training and testing sets *before* any preprocessing steps.
    - **Feature Scaling:** Apply appropriate feature scaling techniques (e.g., StandardScaler, MinMaxScaler) to numerical features before training models.
    - **Categorical Encoding:** Use one-hot encoding or ordinal encoding for categorical features.
    - **Missing Value Imputation:** Handle missing values using imputation techniques like mean, median, or k-NN imputation.
    - **Model Persistence:** Save trained models to disk using `joblib` or `pickle` for later use.

- **Anti-patterns and Code Smells:**
    - **Data Leakage:** Avoid data leakage by preprocessing the entire dataset before splitting into training and testing sets.
    - **Overfitting:** Be cautious of overfitting by using regularization techniques and cross-validation.
    - **Ignoring Data Distributions:**  Always visualize and understand data distributions before applying transformations or choosing models.
    - **Hardcoding Parameters:** Avoid hardcoding parameters directly in the code. Use configuration files or command-line arguments instead.
    - **Ignoring Warnings:** Pay attention to scikit-learn warnings, as they often indicate potential problems with your code or data.

- **State Management:**
    - Use classes to encapsulate state and behavior related to models or data processing steps.
    - Avoid global variables and mutable state whenever possible.
    - Use immutable data structures where appropriate to prevent unintended side effects.

- **Error Handling:**
    - Use try-except blocks to handle exceptions and prevent your application from crashing.
    - Log errors and warnings to a file or logging service.
    - Provide informative error messages to users.
    - Implement retry mechanisms for transient errors.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Vectorization:** Utilize NumPy's vectorized operations to perform calculations on entire arrays instead of looping through individual elements. Use `.values` from pandas DataFrames when passing to scikit-learn.
    - **Memory Optimization:** Minimize memory usage by using appropriate data types (e.g., `int8`, `float32`) and avoiding unnecessary copies of data.
    - **Algorithm Selection:** Choose efficient algorithms that are well-suited for your data and task. For example, use `MiniBatchKMeans` for large datasets.
    - **Parallelization:** Use scikit-learn's built-in support for parallel processing to speed up training and prediction.

- **Memory Management:**
    - Use memory profiling tools to identify memory leaks and optimize memory usage.
    - Release unused memory by deleting variables or using the `gc` module.
    - Use data streaming techniques for large datasets that don't fit into memory.

- **Lazy Loading Strategies:**
    - Load data and models only when they are needed.
    - Use generators or iterators to process large datasets in chunks.

## 4. Security Best Practices

- **Common Vulnerabilities:**
    - **Model Poisoning:** Protect against model poisoning attacks by validating input data and sanitizing user-provided data.
    - **Adversarial Attacks:** Be aware of adversarial attacks that can manipulate model predictions. Consider using techniques like adversarial training to improve model robustness.

- **Input Validation:**
    - Validate all input data to ensure it conforms to the expected format and range.
    - Sanitize user-provided data to prevent injection attacks.

- **Data Protection:**
    - Encrypt sensitive data at rest and in transit.
    - Use access control mechanisms to restrict access to data and models.

## 5. Testing Approaches

- **Unit Testing:**
    - Write unit tests for individual functions and classes.
    - Use mocking to isolate components and test them in isolation.
    - Assert that your code produces the expected output for different inputs.

- **Integration Testing:**
    - Write integration tests to verify that different components work together correctly.
    - Test the interaction between your code and external dependencies.

- **Test Organization:**
    - Organize your tests into a separate directory (e.g., `tests`).
    - Use descriptive names for your test files and functions.

- **Mocking and Stubbing:**
    - Use mocking libraries like `unittest.mock` or `pytest-mock` to create mock objects for testing.
    - Use stubs to replace complex dependencies with simpler implementations.

## 6. Common Pitfalls and Gotchas

- **Data Type Mismatches:** Ensure that the data types of your input features are compatible with the model you are using.
- **Feature Scaling Issues:** Use the correct scaling method for your features, understanding how each scaler behaves.
- **Improper Cross-Validation:** Make sure the cross-validation strategy you choose is appropriate for your dataset and model.
- **Version Compatibility:** Be aware of compatibility issues between different versions of scikit-learn and its dependencies.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **IDE:** Use an IDE like VS Code, PyCharm, or JupyterLab.
    - **Version Control:** Use Git for version control.
    - **Package Manager:** Use pip or Conda for managing dependencies.
    - **Virtual Environments:** Create virtual environments for each project to isolate dependencies.

- **Linting and Formatting:**
    - Use linters like pylint or flake8 to enforce code style and identify potential errors.
    - Use formatters like black or autopep8 to automatically format your code.

- **CI/CD Integration:**
    - Integrate your code with a CI/CD system like Jenkins, Travis CI, or CircleCI.
    - Automatically run tests and linters on every commit.
    - Deploy your application to a production environment automatically.

## General Coding Style
- Use underscores to separate words in non-class names: `n_samples` rather than `nsamples`.
- Avoid multiple statements on one line.
- Use relative imports for references inside scikit-learn (except for unit tests, which should use absolute imports).

By following these best practices, you can write cleaner, more maintainable, and more efficient scikit-learn code.