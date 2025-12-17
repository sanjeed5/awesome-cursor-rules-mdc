---
description: Definitive guidelines for developing robust, performant, and secure applications and data pipelines on Databricks using modern best practices and native services.
globs: **/*
---
# databricks Best Practices

This guide outlines the essential best practices for developing on Databricks. Adhere to these rules to ensure your code is maintainable, performant, and secure.

## 1. Code Organization and Structure

**Leverage Git folders and Databricks Asset Bundles for all projects.** Treat notebooks as version-controlled code, not isolated scripts. Extract reusable logic into Python modules.

*   **Version Control**: Always use Git folders for notebooks and source code.
    ❌ BAD: Storing notebooks directly in Workspace without Git integration.
    ✅ GOOD:
    ```python
    # In a Git-synced notebook (e.g., /Repos/user/my-repo/notebooks/my_pipeline.py)
    # This notebook is version-controlled and can import local modules.
    from ..src.utils import process_data

    df = spark.read.table("raw_data")
    processed_df = process_data(df)
    processed_df.write.mode("overwrite").saveAsTable("processed_data")
    ```

*   **Module Extraction**: For any logic beyond simple notebook orchestration, extract it into Python modules (`.py` files) within your Git repository.
    ❌ BAD:
    ```python
    # In a notebook cell
    def complex_transformation(df):
        # 50+ lines of transformation logic
        return df
    ```
    ✅ GOOD:
    ```python
    # /Repos/user/my-repo/src/transformations.py
    def complex_transformation(df):
        # Modular, testable logic
        return df

    # In a notebook
    from src.transformations import complex_transformation
    df = complex_transformation(spark.read.table("staging"))
    ```

*   **Project Structure with Bundles**: Use Databricks Asset Bundles to define and deploy your entire project (jobs, pipelines, models, notebooks, infrastructure) as a single, versioned unit.
    ✅ GOOD:
    ```yaml
    # databricks-bundle.yml
    bundle:
      name: my-data-pipeline
    resources:
      jobs:
        my_etl_job:
          name: My ETL Job
          tasks:
            - task_key: process_data
              notebook_task:
                notebook_path: ./notebooks/main_pipeline.py
              new_cluster:
                spark_version: "14.3.x-scala2.12"
                node_type_id: "Standard_DS3_v2"
                num_workers: 3
    ```

*   **Pin Dependencies**: Always pin Python package versions in `requirements.txt` for reproducibility, especially for serverless compute.
    ❌ BAD:
    ```
    pandas
    numpy
    ```
    ✅ GOOD:
    ```
    pandas==2.2.3
    numpy==1.26.4
    ```

## 2. Common Patterns and Anti-patterns

**Leverage Databricks-native services for all heavy lifting.** Avoid using Databricks Apps compute for data processing.

*   **Offload Heavy Processing**: Databricks Apps compute is for UI rendering. Use Databricks SQL for ad-hoc queries, Lakeflow Jobs for batch pipelines, and Model Serving for AI inference.
    ❌ BAD:
    ```python
    # In a Databricks App (e.g., Flask app)
    df = spark.sql("SELECT * FROM large_table").toPandas() # Pulls large data to app compute
    processed_data = df.groupby('col').sum() # Heavy processing on app compute
    ```
    ✅ GOOD:
    ```python
    # In a Databricks App
    # Use Databricks SQL for queries
    from databricks.sdk.service.sql import StatementExecutionAPI
    # ... authenticate ...
    statement_execution = StatementExecutionAPI(api_client)
    result = statement_execution.execute_statement(
        warehouse_id="your_sql_warehouse_id",
        statement="SELECT SUM(col) FROM large_table GROUP BY col",
        # ... handle async results ...
    )
    # For batch processing, trigger a Lakeflow Job
    # For AI inference, call a Model Serving endpoint
    ```

*   **Serverless Compute Compatibility**: Ensure data is in Unity Catalog, use Databricks Runtime 14.3+, and avoid JARs.
    ❌ BAD:
    ```python
    # Attempting to use a custom JAR for a data source on serverless
    spark.sparkContext.addJar("s3://my-bucket/custom-connector.jar")
    df = spark.read.format("com.example.CustomSource").load(...)
    ```
    ✅ GOOD:
    ```python
    # Use native ingestion methods for serverless
    # For cloud storage:
    df = spark.readStream.format("cloudFiles") \
        .option("cloudFiles.format", "json") \
        .load("s3://my-bucket/raw_data")

    # For external databases (query federation):
    df = spark.read.table("my_catalog.external_schema.external_table")
    ```

## 3. Performance Considerations

**Optimize resource usage and data access patterns.**

*   **Databricks Apps Startup**: Keep initialization lightweight. Load heavy resources only when needed.
    ❌ BAD:
    ```python
    # app.py
    import large_model_library
    model = large_model_library.load_model("path/to/model") # Blocks startup
    ```
    ✅ GOOD:
    ```python
    # app.py
    model = None
    def get_model():
        nonlocal model
        if model is None:
            import large_model_library
            model = large_model_library.load_model("path/to/model") # Lazy load
        return model
    ```

*   **In-Memory Caching (Apps)**: Cache frequently used data or API responses.
    ✅ GOOD:
    ```python
    from functools import lru_cache

    @lru_cache(maxsize=128)
    def get_cached_data(key):
        # Expensive operation, e.g., query Databricks SQL
        return fetch_data_from_warehouse(key)
    ```

## 4. Common Pitfalls and Gotchas

**Avoid common missteps that lead to instability or security vulnerabilities.**

*   **Graceful Shutdown (Databricks Apps)**: Implement `SIGTERM` handling to shut down within 15 seconds.
    ❌ BAD: No `SIGTERM` handler, app gets `SIGKILL`.
    ✅ GOOD:
    ```python
    import signal
    import sys
    import time

    def signal_handler(signum, frame):
        print("SIGTERM received, initiating graceful shutdown...")
        # Perform cleanup, close connections, etc.
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    # ... your app logic ...
    ```

*   **Logging (Databricks Apps)**: Log to `stdout` and `stderr` only.
    ❌ BAD: `with open("app.log", "a") as f: f.write(...)`
    ✅ GOOD: `print("Log message"); import logging; logging.info("Log message")`

*   **Secrets Management**: Never expose raw secrets. Use `valueFrom` in app config.
    ❌ BAD: `env: MY_API_KEY: "super-secret-value"`
    ✅ GOOD:
    ```yaml
    env:
      MY_API_KEY:
        valueFrom:
          secret:
            name: "my-scope/my-api-key"
    ```

*   **Privileged Operations (Databricks Apps)**: Apps run as non-privileged users.
    ❌ BAD: Attempting `apt-get install` or root access.
    ✅ GOOD: Use Python/Node.js package managers (PyPI, npm).

## 5. Error Handling

**Implement robust error handling and observability.**

*   **Global Exception Handling (Databricks Apps)**: Prevent crashes, return proper HTTP errors.
    ❌ BAD: Uncaught exceptions exposing stack traces.
    ✅ GOOD:
    ```python
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def handle_exception(e):
        # Log the full exception for internal debugging
        app.logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify(error="An unexpected error occurred"), 500
    ```

*   **Observability**: Enable monitoring for Lakeflow Jobs and Pipelines.
    ✅ GOOD: Configure job logging and alerts within Databricks.

## 6. Request/Response Patterns

**Use Databricks SDKs/APIs and secure data exchange.**

*   **Interoperability**: Use the unified Databricks REST API or higher-level SDKs (Python, Java, Go, R) for cross-system integration.
    ❌ BAD: Custom HTTP clients for Databricks API calls.
    ✅ GOOD:
    ```python
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    jobs = w.jobs.list()
    for job in jobs:
        print(job.settings.name)
    ```

*   **Parameterized SQL**: Prevent SQL injection.
    ❌ BAD: `cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")`
    ✅ GOOD: `cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))`

*   **Asynchronous Requests (Apps for Long-Running Operations)**: Avoid synchronous waits.
    ✅ GOOD:
    ```python
    # Initial request to start a job
    job_run = w.jobs.run_now(job_id="my_long_job")
    run_id = job_run.run_id

    # Periodically poll for status
    while w.jobs.get_run(run_id).state.life_cycle_state not in ["TERMINATED", "SKIPPED"]:
        time.sleep(10)
    ```

## 7. Testing Approaches

**Embrace automated testing for reliability.**

*   **Unit Testing**: Unit test extracted Python modules locally or in CI/CD.
    ✅ GOOD:
    ```python
    # test_transformations.py
    import pytest
    from src.transformations import complex_transformation
    from pyspark.sql import SparkSession

    @pytest.fixture(scope="session")
    def spark():
        return SparkSession.builder.appName("pytest-spark").getOrCreate()

    def test_complex_transformation(spark):
        data = [("A", 1), ("B", 2)]
        input_df = spark.createDataFrame(data, ["col1", "col2"])
        output_df = complex_transformation(input_df)
        assert output_df.count() == 2
        # ... more assertions ...
    ```

*   **CI/CD Pipeline**: Implement a full CI/CD pipeline using Databricks Asset Bundles, GitHub Actions, Azure DevOps, etc.
    ✅ GOOD:
    ```yaml
    # .github/workflows/databricks_ci.yml
    name: Databricks CI/CD
    on: [push]
    jobs:
      deploy:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: databricks/setup-cli@v0
          - run: databricks bundle deploy -t production # Deploy using the bundle
            env:
              DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
              DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
    ```