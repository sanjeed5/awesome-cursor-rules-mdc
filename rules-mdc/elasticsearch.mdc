---
description: This rule provides comprehensive best practices for developing with Elasticsearch, covering code organization, performance, security, testing, and common pitfalls, ensuring efficient and maintainable Elasticsearch applications. These practices apply broadly across languages interacting with Elasticsearch.
globs: *.py,*.js,*.java,*.go,*.ts,*.kt,*.scala,*.rb,*.php,*.rs,*.c,*.cpp,*.cs,*.swift,*.m,*.h
---
- # Elasticsearch Library Best Practices

  This document provides comprehensive guidelines for developing with Elasticsearch, covering code organization, common patterns, performance considerations, security best practices, testing approaches, common pitfalls, and tooling.

- ## 1. Code Organization and Structure

  - ### Directory Structure

    - **Project Root:** Contains the main application code, configuration files, and build scripts.
    - **src/:** Source code for your application. This directory is further subdivided:
      - **config/:** Configuration files for Elasticsearch connections and settings.
      - **mappings/:** JSON files defining Elasticsearch index mappings.
      - **queries/:** Stored Elasticsearch queries for reuse.
      - **models/:** Data models representing Elasticsearch documents.
      - **utils/:** Utility functions for interacting with Elasticsearch.
      - **scripts/:** Scripts for indexing, reindexing, and data migration.
    - **tests/:** Unit and integration tests.
    - **docs/:** Documentation for your application and Elasticsearch integration.
    - **data/:** Sample data for testing and development.
    - **.cursor/:** Stores cursor rule files if using Cursor IDE.

  - ### File Naming Conventions

    - **Configuration Files:** `elasticsearch.config.json`, `index.mapping.json`
    - **Data Models:** `user.model.js`, `product.model.py`
    - **Utility Functions:** `elasticsearch.utils.go`, `query_builder.rb`
    - **Test Files:** `user.model.test.js`, `elasticsearch.utils_test.py`

  - ### Module Organization

    - Divide your code into logical modules, such as:
      - **elasticsearch_client:** Handles the Elasticsearch client initialization and connection management.
      - **index_management:** Manages index creation, deletion, and mapping updates.
      - **query_builder:** Provides functions for building complex Elasticsearch queries.
      - **data_ingestion:** Handles data ingestion and indexing processes.
      - **search_service:** Implements the search logic and result processing.

  - ### Component Architecture

    - Use a layered architecture to separate concerns:
      - **Presentation Layer:** Handles user input and displays search results.
      - **Application Layer:** Orchestrates the interaction between the presentation and domain layers.
      - **Domain Layer:** Contains the core business logic and data models.
      - **Infrastructure Layer:** Provides access to external services like Elasticsearch.

  - ### Code Splitting

    -  If your application handles large datasets or complex queries, consider code splitting:
       -  Split large mapping files into smaller, manageable chunks. Use `@file` directives if using Cursor to include them in the primary rule file.
       -  Implement lazy loading for infrequently used features.
       -  Optimize queries to minimize the amount of data transferred.

- ## 2. Common Patterns and Anti-patterns

  - ### Design Patterns

    - **Repository Pattern:** Abstract data access logic behind a repository interface.
    - **Factory Pattern:** Create Elasticsearch clients and query builders using factories.
    - **Builder Pattern:** Construct complex Elasticsearch queries step-by-step.
    - **Observer Pattern:** Notify subscribers of index updates or data changes.

  - ### Recommended Approaches

    - **Bulk Indexing:** Use the bulk API for efficient indexing of large datasets.
    - **Scroll API:** Use the scroll API for retrieving large result sets.
    - **Asynchronous Operations:** Use asynchronous operations for non-blocking Elasticsearch interactions.

  - ### Anti-patterns and Code Smells

    - **Over-sharding:** Avoid creating too many shards, which can lead to performance issues.
    - **Ignoring Errors:** Always handle Elasticsearch errors and log them appropriately.
    - **Hardcoding Queries:** Avoid hardcoding queries directly in your code. Use query builders or stored queries.
    - **Excessive Data Retrieval:** Only retrieve the fields you need from Elasticsearch.

  - ### State Management

    - Keep track of the index state (e.g., creation date, mapping version).
    - Use version control for index mappings and queries.
    - Implement rollback mechanisms for data migration.

  - ### Error Handling

    - Use try-except blocks to catch Elasticsearch exceptions.
    - Log errors with sufficient context for debugging.
    - Implement retry mechanisms for transient errors.
    - Implement circuit breaker pattern to prevent cascading failures.

- ## 3. Performance Considerations

  - ### Optimization Techniques

    - **Mapping Optimization:**
      - Use the appropriate data types for your fields.
      - Use `keyword` type for fields that are used for filtering or sorting.
      - Use `text` type for fields that are used for full-text search.
      - Disable indexing for fields that are only used for retrieval (`"index": false`).
      - Consider using `multi-fields` for different types of searches (e.g., keyword and text).
    - **Query Optimization:**
      - Use the `filter` context for queries that do not affect scoring.
      - Use the `bool` query to combine multiple conditions effectively.
      - Use the `match` query for full-text searches.
      - Utilize caching mechanisms for frequently executed queries.
    - **Indexing Optimization:**
      - Use the bulk API for indexing large amounts of data.
      - Increase the refresh interval for less frequent updates.
      - Optimize shard sizes based on your data volume.
      - Avoid over-sharding.

  - ### Memory Management

    - Monitor the Elasticsearch heap usage.
    - Configure the JVM heap size appropriately.
    - Avoid memory leaks in your application code.

  - ### Rendering Optimization (If Applicable)

    - Implement pagination for large result sets.
    - Use virtualization techniques for rendering large lists.
    - Optimize the rendering performance of your UI components.

  - ### Bundle Size Optimization

    - Minimize the size of your application's bundle.
    - Use code splitting to load only the necessary modules.
    - Optimize your dependencies and remove unused code.

  - ### Lazy Loading

    - Implement lazy loading for infrequently used features.
    - Load data on demand as needed.
    - Use asynchronous operations for loading data.

- ## 4. Security Best Practices

  - ### Common Vulnerabilities

    - **Injection Attacks:** Prevent injection attacks by validating user input and sanitizing data.
    - **Unauthorized Access:** Implement proper authentication and authorization mechanisms.
    - **Data Breaches:** Protect sensitive data by encrypting it at rest and in transit.

  - ### Input Validation

    - Validate user input to prevent injection attacks and data corruption.
    - Sanitize data before indexing it into Elasticsearch.
    - Use regular expressions or validation libraries to enforce data constraints.

  - ### Authentication and Authorization

    - Use Elasticsearch's built-in security features or a third-party plugin for authentication.
    - Implement role-based access control (RBAC) to restrict access to sensitive data.
    - Use API keys or tokens for secure API communication.

  - ### Data Protection

    - Encrypt sensitive data at rest using Elasticsearch's encryption features.
    - Encrypt data in transit using HTTPS.
    - Mask or redact sensitive data in logs and error messages.

  - ### Secure API Communication

    - Use HTTPS for all API communication with Elasticsearch.
    - Validate the server certificate to prevent man-in-the-middle attacks.
    - Use secure protocols like TLS 1.3 or higher.

- ## 5. Testing Approaches

  - ### Unit Testing

    - Unit test individual components in isolation.
    - Mock Elasticsearch dependencies to control test behavior.
    - Use assertion libraries to verify the expected results.

  - ### Integration Testing

    - Integrate test different components to verify their interaction.
    - Use a test Elasticsearch instance for integration tests.
    - Verify that data is correctly indexed and retrieved.

  - ### End-to-End Testing

    - Test the entire application from end to end.
    - Use automated testing frameworks like Selenium or Cypress.
    - Verify that the application functions correctly in a real-world environment.

  - ### Test Organization

    - Organize your tests into logical categories.
    - Use descriptive names for your test cases.
    - Keep your tests concise and focused.

  - ### Mocking and Stubbing

    - Use mocking libraries to create mock Elasticsearch clients and responses.
    - Use stubbing to simulate different Elasticsearch scenarios.
    - Verify that your code interacts with Elasticsearch as expected.

- ## 6. Common Pitfalls and Gotchas

  - ### Frequent Mistakes

    - Not understanding Elasticsearch's query DSL.
    - Not configuring the Elasticsearch cluster properly.
    - Not monitoring Elasticsearch performance.

  - ### Edge Cases

    - Handling large result sets.
    - Dealing with complex data types.
    - Managing index mappings and updates.

  - ### Version-Specific Issues

    - Be aware of breaking changes between Elasticsearch versions.
    - Test your application thoroughly when upgrading Elasticsearch.
    - Consult the Elasticsearch documentation for version-specific information.

  - ### Compatibility Concerns

    - Ensure compatibility between Elasticsearch and your application's programming language.
    - Use compatible versions of Elasticsearch client libraries.

  - ### Debugging Strategies

    - Use Elasticsearch's logging features to troubleshoot issues.
    - Use the Elasticsearch API to inspect the cluster state.
    - Use debugging tools to step through your code.

- ## 7. Tooling and Environment

  - ### Recommended Tools

    - **Elasticsearch:** The core search and analytics engine.
    - **Kibana:** A visualization and exploration tool for Elasticsearch data.
    - **Logstash:** A data processing pipeline for ingesting data into Elasticsearch.
    - **Elasticsearch Head/Cerebro:** Cluster management and monitoring tools.

  - ### Build Configuration

    - Use a build tool like Maven, Gradle, or npm to manage dependencies.
    - Configure your build to package your application and its dependencies.

  - ### Linting and Formatting

    - Use a linter like ESLint or Pylint to enforce code style guidelines.
    - Use a formatter like Prettier or Black to automatically format your code.

  - ### Deployment

    - Deploy your Elasticsearch cluster to a cloud platform like AWS, Azure, or GCP.
    - Use containerization technologies like Docker and Kubernetes.
    - Automate your deployment process using tools like Ansible or Terraform.

  - ### CI/CD

    - Integrate your application with a CI/CD pipeline.
    - Automate testing, building, and deployment processes.
    - Use tools like Jenkins, GitLab CI, or CircleCI.

- ## Additional Best Practices

  - **Monitoring:** Implement comprehensive monitoring of Elasticsearch cluster health, resource utilization, and query performance using tools like Prometheus, Grafana, and Elasticsearch's built-in monitoring features.
  - **Logging:** Configure detailed logging to capture important events, errors, and performance metrics for troubleshooting and analysis. Use structured logging formats like JSON for easier parsing and analysis.
  - **Backup and Recovery:** Implement regular backups of Elasticsearch data and configurations to prevent data loss. Test the recovery process to ensure it works as expected.
  - **Capacity Planning:** Regularly assess the capacity requirements of your Elasticsearch cluster based on data growth, query load, and indexing performance. Scale your cluster accordingly to maintain optimal performance.
  - **Security Auditing:** Conduct regular security audits to identify and address potential vulnerabilities. Review access controls, network configurations, and data encryption settings.
  - **Stay Updated:** Keep your Elasticsearch cluster and client libraries up to date with the latest security patches and bug fixes. Follow the Elasticsearch community for updates and best practices.

  By following these best practices, you can build robust, scalable, and secure applications with Elasticsearch.