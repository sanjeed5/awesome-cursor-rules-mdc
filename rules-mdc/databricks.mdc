---
description: This rule file provides comprehensive best practices for Databricks development, covering code organization, performance, security, testing, and common pitfalls. Adhering to these guidelines promotes maintainable, efficient, and secure Databricks applications.
globs: **/*.{py,sql,ipynb,scala,r}
---
- **Follow these best practices to ensure maintainable, efficient, and secure Databricks applications.**

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices

*   **Project Root:**
    *   `src/`: Contains source code.
        *   `jobs/`: Contains Databricks job definitions (Python, Scala, or notebooks).
        *   `libraries/`: Contains reusable code modules (Python packages, JAR files).
        *   `notebooks/`: Contains Databricks notebooks. Structure notebooks by function or data domain.
        *   `sql/`: Contains SQL scripts for data transformations and queries.
    *   `tests/`: Contains unit, integration, and end-to-end tests.
    *   `config/`: Contains configuration files for different environments (dev, staging, prod).
    *   `data/`: (Optional) Small sample datasets for development and testing.
    *   `docs/`: Project documentation.
    *   `scripts/`: Deployment and utility scripts.
    *   `.databricks/`: Databricks CLI configuration.
    *   `requirements.txt` or `pyproject.toml`: Python dependency management.
    *   `build.sbt`: Scala build definition (if applicable).
    *   `README.md`: Project overview and instructions.
*   **Example:**


my_databricks_project/
├── src/
│   ├── jobs/
│   │   ├── daily_etl.py
│   │   └── model_training.py
│   ├── libraries/
│   │   ├── data_utils/
│   │   │   ├── __init__.py
│   │   │   ├── data_cleaning.py
│   │   │   └── data_validation.py
│   │   └── feature_engineering/
│   ├── notebooks/
│   │   ├── data_exploration.ipynb
│   │   ├── model_evaluation.ipynb
│   │   └── reporting.ipynb
│   └── sql/
│       ├── create_tables.sql
│       └── transform_data.sql
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── config/
│   ├── dev.conf
│   ├── staging.conf
│   └── prod.conf
├── data/
├── docs/
├── scripts/
├── .databricks/
├── requirements.txt
├── build.sbt
└── README.md


### 1.2. File Naming Conventions

*   **Python scripts:** `lowercase_with_underscores.py`
*   **SQL scripts:** `lowercase_with_underscores.sql`
*   **Notebooks:** `descriptive-name.ipynb` (use hyphens for readability)
*   **Configuration files:** `environment.conf` (e.g., `dev.conf`, `prod.conf`)
*   **Data files:** `descriptive_name.csv`, `descriptive_name.parquet`
*   **Scala files:** `PascalCaseName.scala`

### 1.3. Module Organization

*   **Separate Concerns:** Group related functions and classes into modules.
*   **Avoid Circular Dependencies:** Ensure modules don't depend on each other in a circular way.
*   **Use Packages:** Organize modules into packages for larger projects.
*   **`__init__.py`:** Use `__init__.py` files to define packages in Python.
*   **Relative Imports:** Use relative imports (`from . import module`) within packages.

### 1.4. Component Architecture

*   **Layered Architecture:** Consider a layered architecture with separate layers for data access, business logic, and presentation (notebooks).
*   **Microservices:** For complex applications, consider breaking them down into smaller, independent microservices.
*   **Data Lakehouse Architecture:** Leverage the Databricks Lakehouse architecture with Delta Lake for reliable and performant data storage.
*   **Modular Notebooks:** Design notebooks as modular components with clear inputs and outputs.

### 1.5. Code Splitting Strategies

*   **Functions:** Break down large functions into smaller, reusable functions.
*   **Modules:** Group related functions and classes into modules.
*   **Libraries:** Create reusable libraries for common tasks.
*   **Notebook Includes:** Use `%run` to include code from other notebooks.
*   **DBUtils:** Use `dbutils.fs.cp` and other DBUtils functions for code reuse accross notebooks.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Data Lakehouse Pattern:** Use Delta Lake for ACID transactions, schema enforcement, and data versioning.
*   **ETL/ELT Pattern:** Design data pipelines using ETL or ELT approaches.
*   **Model-View-Controller (MVC):** (For complex applications) Separate data, logic, and presentation.
*   **Singleton Pattern:** (Use carefully) For managing global resources.
*   **Factory Pattern:** For creating objects in a flexible and decoupled way.

### 2.2. Recommended Approaches for Common Tasks

*   **Data Ingestion:** Use Auto Loader for incremental data loading from cloud storage.
*   **Data Transformation:** Use Spark DataFrames and SQL for data transformations.
*   **Data Validation:** Implement data quality checks using Delta Lake constraints and custom validation functions.
*   **Model Training:** Use MLflow for experiment tracking and model management.
*   **Model Deployment:** Use Databricks Model Serving or MLflow Model Registry for model deployment.
*   **Job Orchestration:** Use Databricks Jobs for scheduling and managing data pipelines.
*   **Configuration Management:** Use Databricks secrets for storing sensitive information.

### 2.3. Anti-patterns and Code Smells

*   **Hardcoding Values:** Avoid hardcoding values; use configuration files or environment variables.
*   **Large Notebooks:** Break down large notebooks into smaller, modular notebooks.
*   **Copy-Pasting Code:** Avoid copy-pasting code; create reusable functions and modules.
*   **Ignoring Errors:** Handle errors gracefully and log them appropriately.
*   **Inefficient Data Transformations:** Optimize data transformations using Spark best practices.
*   **Over-commenting:** Comments should explain the *why* not the *what*.  Code should be self-documenting as much as possible.
*   **Ignoring Code Style:** Follow a consistent code style (e.g., PEP 8 for Python).
*   **Storing large dataframes in notebook's memory:** Instead store data in Delta tables or cloud storage.

### 2.4. State Management

*   **Stateless Transformations:** Design data transformations to be stateless whenever possible.
*   **Delta Lake:** Use Delta Lake for managing state in data pipelines.
*   **Databricks Secrets:** Use Databricks secrets for managing sensitive information such as API keys and passwords.
*   **Configuration Files:** Externalize configurable data in configuration files.

### 2.5. Error Handling

*   **`try-except` Blocks:** Use `try-except` blocks to handle exceptions gracefully.
*   **Logging:** Log errors and warnings to a central logging system.
*   **Custom Exceptions:** Define custom exceptions for specific error conditions.
*   **Retry Logic:** Implement retry logic for transient errors.
*   **Alerting:** Set up alerting for critical errors.
*   **DBUtils.notebook.exit:** Use `dbutils.notebook.exit()` to gracefully exit notebooks.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Partitioning:** Partition data based on common query patterns.
*   **Bucketing:** Bucket data for faster joins and aggregations.
*   **Caching:** Cache frequently accessed data using `spark.cache()` or `spark.persist()`.
*   **Broadcast Joins:** Use broadcast joins for small tables.
*   **Predicate Pushdown:** Push down filters to the data source.
*   **Data Skipping:** Use Delta Lake data skipping to skip irrelevant data files.
*   **Optimize Writes:** Optimize Delta Lake write performance by controlling file size and number of partitions.
*   **Avoid User-Defined Functions (UDFs):**  Prefer Spark built-in functions. If UDF is unavoidable, explore vectorization.
*   **Optimize cluster configuration:** Choose the correct driver and worker instance types.  Right size your cluster.
*   **Avoid shuffling data:** Minimize data movement across the network.

### 3.2. Memory Management

*   **Avoid Large DataFrames:** Avoid loading large DataFrames into memory at once. Use iterative processing or pagination.
*   **Garbage Collection:** Monitor garbage collection and tune JVM settings if necessary.
*   **Spark Memory Configuration:** Configure Spark memory settings (e.g., `spark.driver.memory`, `spark.executor.memory`).
*   **Off-Heap Memory:** Consider using off-heap memory for large datasets.
*   **Delta Lake Vacuum:** Regularly vacuum Delta Lake tables to remove old versions and reclaim storage space.

### 3.3. Rendering Optimization (if applicable)

*   **Limit Data Display:** Limit the amount of data displayed in notebooks to avoid performance issues.
*   **Use Visualizations:** Use visualizations to summarize large datasets.
*   **Optimize Plotting Libraries:** Optimize plotting library settings for performance.

### 3.4. Bundle Size Optimization (for custom web applications using Databricks)

*   **Code Splitting:** Split code into smaller chunks for lazy loading.
*   **Tree Shaking:** Remove unused code during the build process.
*   **Minification:** Minify code to reduce file size.
*   **Compression:** Compress static assets (CSS, JavaScript, images).

### 3.5. Lazy Loading

*   **Lazy Data Loading:** Load data only when it's needed.
*   **Spark Lazy Evaluation:** Utilize Spark's lazy evaluation to defer computations.
*   **Dynamic Imports:** Use dynamic imports to load modules only when they're needed.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Code Injection:** Prevent code injection vulnerabilities by validating user inputs.
*   **SQL Injection:** Use parameterized queries to prevent SQL injection attacks.
*   **Cross-Site Scripting (XSS):** (If using Databricks for web applications) Prevent XSS vulnerabilities by sanitizing user inputs.
*   **Broken Authentication:** Use strong authentication and authorization mechanisms.
*   **Sensitive Data Exposure:** Protect sensitive data by encrypting it at rest and in transit.

### 4.2. Input Validation

*   **Data Types:** Validate data types to prevent type errors.
*   **Range Checks:** Validate that values are within acceptable ranges.
*   **Regular Expressions:** Use regular expressions to validate data formats.
*   **Allow Lists:** Use allow lists to restrict input to known good values.

### 4.3. Authentication and Authorization

*   **Databricks Access Control:** Use Databricks access control to manage user permissions.
*   **Unity Catalog:** Use Unity Catalog for centralized data governance and access control.
*   **Service Principals:** Use service principals for automated access to Databricks resources.
*   **Multi-Factor Authentication (MFA):** Enforce MFA for user accounts.

### 4.4. Data Protection

*   **Encryption:** Encrypt sensitive data at rest and in transit.
*   **Data Masking:** Mask sensitive data in logs and reports.
*   **Data Redaction:** Redact sensitive data from data files.
*   **Data Auditing:** Enable data auditing to track data access and modifications.
*   **Row-level and Column-level Security:** Implement fine grained access controls through Unity Catalog.

### 4.5. Secure API Communication

*   **HTTPS:** Use HTTPS for all API communication.
*   **API Keys:** Use API keys for authentication.
*   **OAuth 2.0:** Use OAuth 2.0 for delegated authorization.
*   **Rate Limiting:** Implement rate limiting to prevent abuse.
*   **Input Sanitization:** Sanitize all API input parameters to prevent injection attacks.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Frameworks:** Use testing frameworks like `pytest` for Python and `ScalaTest` for Scala.
*   **Test Coverage:** Aim for high test coverage.
*   **Test-Driven Development (TDD):** Consider using TDD to write tests before code.
*   **Mocking and Stubbing:** Use mocking and stubbing to isolate units of code.
*   **Property-based Testing:** Generate a wide range of test inputs, which is useful when you need a broad range of possible data permutations to test your code.

### 5.2. Integration Testing

*   **Test Data Pipelines:** Test the integration of different components in data pipelines.
*   **Test Data Quality:** Test data quality by validating data transformations and aggregations.
*   **Test External Systems:** Test the integration with external systems.

### 5.3. End-to-End Testing

*   **Test Full Workflows:** Test full workflows from data ingestion to reporting.
*   **Test User Interfaces:** (If applicable) Test user interfaces to ensure they function correctly.
*   **Automate Testing:** Automate end-to-end tests using CI/CD pipelines.

### 5.4. Test Organization

*   **Separate Test Directories:** Create separate directories for unit, integration, and end-to-end tests.
*   **Test Naming Conventions:** Use clear and consistent test naming conventions.
*   **Test Suites:** Organize tests into test suites.

### 5.5. Mocking and Stubbing

*   **Mock External Dependencies:** Mock external dependencies such as databases and APIs.
*   **Stub Functions:** Stub functions to control their behavior during testing.
*   **Mock DataFrames:** Create mock DataFrames for testing data transformations.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Not Understanding Spark's Execution Model:** Understanding lazy evaluation and transformations vs. actions is crucial.
*   **Inefficient Data Partitioning:** Choosing incorrect partitioning can lead to performance bottlenecks.
*   **Not Using Delta Lake Features:** Failing to leverage Delta Lake features such as ACID transactions and data versioning.
*   **Over-reliance on Notebooks:** Putting all code into notebooks instead of creating reusable modules.
*   **Ignoring Security Best Practices:** Failing to implement proper security measures.

### 6.2. Edge Cases

*   **Handling Null Values:** Be aware of how Spark handles null values in data transformations.
*   **Time Zone Issues:** Handle time zone conversions carefully.
*   **Data Skew:** Address data skew by repartitioning or salting data.
*   **Large Files:** Handle large files efficiently by using streaming or chunking.

### 6.3. Version-Specific Issues

*   **Spark Version Compatibility:** Be aware of compatibility issues between different Spark versions.
*   **Databricks Runtime Version:** Be aware of the Databricks Runtime version and its limitations.
*   **Library Version Conflicts:** Manage library version conflicts using dependency management tools.

### 6.4. Compatibility Concerns

*   **Integration with Cloud Services:** Be aware of compatibility issues when integrating with cloud services such as AWS, Azure, and GCP.
*   **Integration with External Databases:** Be aware of compatibility issues when integrating with external databases such as MySQL, PostgreSQL, and SQL Server.
*   **Integration with BI Tools:** Be aware of compatibility issues when integrating with BI tools such as Tableau, Power BI, and Looker.

### 6.5. Debugging Strategies

*   **Spark UI:** Use the Spark UI to monitor Spark jobs and identify performance bottlenecks.
*   **Logging:** Use logging to track the execution of code and identify errors.
*   **Debugging Tools:** Use debugging tools such as `pdb` for Python and the IntelliJ debugger for Scala.
*   **Explain Plans:** Use explain plans to understand how Spark executes queries.
*   **Delta Lake History:** Use Delta Lake history to track data changes and debug data issues.
*   **Databricks Profiler:** Use Databricks Profiler to analyze code execution and identify performance bottlenecks.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **Databricks Notebooks:** Use Databricks notebooks for interactive development and exploration.
*   **Databricks Connect:** Use Databricks Connect to connect to Databricks clusters from local IDEs.
*   **VS Code:** Use VS Code with the Databricks extension for code editing and debugging.
*   **IntelliJ IDEA:** Use IntelliJ IDEA for Scala development.
*   **Jupyter Notebook:** Use Jupyter Notebook for Python development.

### 7.2. Build Configuration

*   **Maven or SBT (Scala):** Use Maven or SBT for building Scala projects.
*   **`requirements.txt` (Python):** Use `requirements.txt` to manage Python dependencies.
*    **`pyproject.toml` (Python):** Use `pyproject.toml` with poetry or other modern tools.
*   **Databricks CLI:** Use the Databricks CLI to manage Databricks resources.
*   **Workspace Bundles:** Use Databricks Workspace Bundles to define and deploy infrastructure and code.

### 7.3. Linting and Formatting

*   **PEP 8 (Python):** Follow PEP 8 for Python code style.
*   **`flake8` (Python):** Use `flake8` for linting Python code.
*   **`black` (Python):** Use `black` for formatting Python code.
*   **Scalafmt (Scala):** Use Scalafmt for formatting Scala code.

### 7.4. Deployment

*   **Databricks Jobs:** Use Databricks Jobs for scheduling and managing data pipelines.
*   **Databricks Workflows:** Use Databricks Workflows to orchestrate complex data pipelines.
*   **Databricks Model Serving:** Use Databricks Model Serving to deploy machine learning models.
*   **Terraform:** Use Terraform to manage Databricks infrastructure as code.
*   **Workspace Bundles:** Use Databricks Workspace Bundles for reproducible deployments.

### 7.5. CI/CD Integration

*   **GitHub Actions:** Use GitHub Actions for CI/CD pipelines.
*   **Azure DevOps:** Use Azure DevOps for CI/CD pipelines.
*   **Jenkins:** Use Jenkins for CI/CD pipelines.
*   **Databricks Repos:** Use Databricks Repos for version control and collaboration.

By following these best practices, you can build maintainable, efficient, and secure Databricks applications that deliver value to your organization. Remember to adapt these guidelines to your specific project requirements and coding style.