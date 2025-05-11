---
description: This rule outlines best practices for using the pandas library in Python, covering code style, performance, data handling, and testing. It aims to promote maintainable, efficient, and robust data analysis workflows.
globs: *.py
---
# Pandas Best Practices

This document provides guidelines for writing high-quality pandas code, covering various aspects from code style to performance optimization.

## 1. Code Organization and Structure

- **Directory Structure:**
    - Organize your project with a clear directory structure.
    - Use separate directories for data, scripts, modules, tests, and documentation.
    - Example:
        
        project_root/
        ├── data/
        │   ├── raw/
        │   └── processed/
        ├── src/
        │   ├── modules/
        │   │   ├── data_cleaning.py
        │   │   ├── feature_engineering.py
        │   │   └── ...
        │   ├── main.py
        ├── tests/
        │   ├── unit/
        │   ├── integration/
        │   └── ...
        ├── docs/
        ├── notebooks/
        ├── requirements.txt
        └── ...
        

- **File Naming Conventions:**
    - Use descriptive and consistent file names.
    - Use snake_case for Python files and variables.
    - Example: `data_processing.py`, `load_data.py`

- **Module Organization:**
    - Break down your code into reusable modules.
    - Each module should focus on a specific task or functionality.
    - Use clear and concise function and class names within modules.
    - Follow the Single Responsibility Principle (SRP).

- **Component Architecture:**
    - Design your pandas-based applications with a clear component architecture.
    - Separate data loading, preprocessing, analysis, and visualization into distinct components.
    - Use classes to encapsulate related functionality and data.

- **Code Splitting Strategies:**
    - Split large pandas operations into smaller, more manageable steps.
    - Use functions or methods to encapsulate these steps.
    - This improves readability and makes debugging easier.

## 2. Common Patterns and Anti-patterns

- **Design Patterns Specific to Pandas:**
    - **Chain of Responsibility:** Use method chaining to perform a sequence of operations on a DataFrame.
    - **Strategy Pattern:** Implement different data processing strategies based on input parameters.
    - **Factory Pattern:** Use factory functions to create DataFrames or Series from various sources.

- **Recommended Approaches for Common Tasks:**
    - **Data Loading:** Use `pd.read_csv()`, `pd.read_excel()`, `pd.read_sql()` to load data from different sources.
    - **Data Cleaning:** Use `dropna()`, `fillna()`, `replace()` to clean missing or incorrect data.
    - **Data Transformation:** Use `apply()`, `map()`, `groupby()`, `pivot_table()` to transform data.
    - **Data Aggregation:** Use `groupby()`, `agg()`, `transform()` to aggregate data.
    - **Data Visualization:** Use `matplotlib`, `seaborn`, or `plotly` to visualize data.

- **Anti-patterns and Code Smells to Avoid:**
    - **Iterating over DataFrames with `iterrows()` or `itertuples()`:** These are slow; prefer vectorized operations.
    - **Chained Indexing:** Avoid using chained indexing (e.g., `df['A']['B']`) as it can lead to unexpected behavior. Use `.loc` or `.iloc` instead.
    - **Ignoring Data Types:** Always be aware of your data types and convert them appropriately using `astype()`.
    - **Writing loops to filter/process the data:** Vectorize these operations to avoid performance issues.
    - **Modifying DataFrame inplace:** Creating copies is a safer approach.

- **State Management Best Practices:**
    - Avoid modifying DataFrames in place unless absolutely necessary.
    - Make copies of DataFrames using `.copy()` to avoid unintended side effects.
    - Use functional programming principles where possible to minimize state changes.

- **Error Handling Patterns:**
    - Use `try-except` blocks to handle potential errors during data loading, processing, or analysis.
    - Log errors and warnings using the `logging` module.
    - Raise exceptions when necessary to signal errors to the calling code.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Vectorization:** Use vectorized operations instead of loops whenever possible. Pandas is optimized for vectorized operations.
    - **Cython:** Consider using Cython to optimize performance-critical parts of your code.
    - **Numba:** Use Numba to JIT-compile NumPy and pandas code for improved performance.
    - **Dask:** Use Dask for parallel processing of large datasets that don't fit in memory.
    - **Parquet or Feather:** Use Parquet or Feather file formats for efficient data storage and retrieval.
    - **Categorical Data:** Use categorical data types for columns with a limited number of unique values.

- **Memory Management:**
    - **Data Types:** Choose appropriate data types to minimize memory usage (e.g., `int8`, `float32` instead of `int64`, `float64`).
    - **Chunking:** Load large datasets in chunks to avoid memory errors.
    - **Garbage Collection:** Use `gc.collect()` to explicitly release memory.
    - **Sparse Data:** Use sparse data structures for data with many missing values.

## 4. Security Best Practices

- **Common Vulnerabilities and How to Prevent Them:**
    - **SQL Injection:** When reading data from SQL databases, use parameterized queries or ORM frameworks to prevent SQL injection attacks.
    - **CSV Injection:** Be cautious when loading CSV files from untrusted sources, as they may contain malicious formulas that can execute arbitrary code.
    - **Arbitrary Code Execution:** Avoid using `eval()` or `exec()` on data loaded from untrusted sources, as this can lead to arbitrary code execution.

- **Input Validation:**
    - Validate user inputs to prevent malicious data from entering your pandas workflows.
    - Use regular expressions or custom validation functions to check the format and content of inputs.

- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use access control mechanisms to restrict access to sensitive data.
    - Anonymize or pseudonymize data when possible to protect privacy.

## 5. Testing Approaches

- **Unit Testing Strategies:**
    - Write unit tests for individual functions and classes.
    - Use `pytest` or `unittest` for writing and running tests.
    - Test edge cases and boundary conditions.
    - Use assert statements to verify the correctness of your code.

- **Integration Testing:**
    - Write integration tests to verify the interaction between different modules or components.
    - Test the end-to-end functionality of your pandas-based applications.

- **Test Organization:**
    - Organize your tests in a separate `tests` directory.
    - Use a clear naming convention for your test files and functions.
    - Example: `test_data_cleaning.py`, `test_load_data()`

- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate units of code during testing.
    - Use the `unittest.mock` module or third-party libraries like `pytest-mock`.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes Developers Make:**
    - **Forgetting to set the index properly:** This can lead to performance issues when joining or merging DataFrames.
    - **Incorrectly handling missing data:** Be aware of how missing data is represented in your DataFrames and handle it appropriately.
    - **Not understanding the difference between `.loc` and `.iloc`:** These methods are used for different types of indexing and can lead to unexpected results if used incorrectly.

- **Edge Cases to Be Aware Of:**
    - **Empty DataFrames:** Handle the case where a DataFrame is empty.
    - **DataFrames with duplicate indices:** Be aware of how pandas handles DataFrames with duplicate indices.

- **Debugging Strategies:**
    - Use the `print()` function or the `logging` module to debug your code.
    - Use a debugger to step through your code and inspect variables.
    - Use the `pdb` module for interactive debugging.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - **Jupyter Notebook/Lab:** For interactive data exploration and analysis.
    - **VS Code with Python extension:** For code editing, debugging, and testing.
    - **PyCharm:** A full-featured IDE for Python development.

- **Linting and Formatting:**
    - Use `flake8` to lint your code and identify potential issues.
    - Use `black` to automatically format your code according to PEP 8.
    - Use `isort` to automatically sort your imports.
    - Integrate these tools into your pre-commit hooks to ensure consistent code style.

- **CI/CD Integration:**
    - Use CI/CD pipelines to automate the testing and deployment of your pandas-based applications.
    - Integrate your CI/CD pipelines with your version control system (e.g., GitHub, GitLab, Bitbucket).
    - Use Docker to containerize your applications for consistent deployment.

This comprehensive guide covers the key aspects of pandas best practices and coding standards. By following these guidelines, you can write more maintainable, efficient, and robust pandas code.