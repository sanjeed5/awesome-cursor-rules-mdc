---
description: This rule provides best practices and coding standards for developing applications using AWS DynamoDB. It covers aspects like schema design, performance optimization, security, and testing.
globs: *.js,*.ts,*.jsx,*.tsx,*.py
---
- **Minimize the number of tables; prefer single-table design.** Having fewer tables keeps things more scalable, requires less permissions management, and reduces overhead for your DynamoDB application. Consider using a single table with appropriate use of primary and secondary indexes.
- **Identify query patterns before schema design.** Understand the application's access patterns before designing the DynamoDB schema. Knowing how the data will be queried is crucial for optimizing performance and cost.
- **Use sort order for related items and distribute queries to avoid hot spots.** Group related items together by utilizing sort keys and design data keys to distribute traffic evenly across partitions.
- **Prefer queries over scans for efficiency.** DynamoDB queries are more efficient and less costly than scan operations. Always aim to use queries that filter based on partition and sort keys.
- **Use caching to reduce read/write costs.** Implement caching strategies, such as AWS ElastiCache or DynamoDB Accelerator (DAX), to minimize DynamoDB read/write costs.  Use appropriate invalidation logic.
- **Validate data integrity in the application layer.** Since DynamoDB doesn't enforce relationship data or integrity constraints, it's essential to perform strict data validations in the application or business layer.

## 1. Code Organization and Structure

- **Directory Structure Best Practices:**
    - `src/`: Contains the application's source code.
    - `src/models/`: Defines data models representing DynamoDB entities.
    - `src/services/`: Implements business logic and interacts with DynamoDB.
    - `src/repositories/`: Handles DynamoDB data access operations.
    - `src/utils/`: Contains utility functions (e.g., data validation, error handling).
    - `tests/`: Includes unit, integration, and end-to-end tests.
    - `config/`: Stores configuration files (e.g., DynamoDB table names, AWS region).
- **File Naming Conventions:**
    - Use descriptive names for files and directories.
    - Model files: `user.model.js` or `user.model.ts`
    - Service files: `user.service.js` or `user.service.ts`
    - Repository files: `user.repository.js` or `user.repository.ts`
    - Test files: `user.service.test.js` or `user.service.test.ts`
- **Module Organization Best Practices:**
    - Encapsulate DynamoDB interactions within dedicated modules (e.g., repositories).
    - Use dependency injection to manage dependencies (e.g., DynamoDB client).
    - Follow the single responsibility principle for modules.
- **Component Architecture Recommendations:**
    - Adopt a layered architecture (e.g., presentation, application, domain, infrastructure).
    - Decouple components to improve maintainability and testability.
    - Use interfaces to define contracts between components.
- **Code Splitting Strategies:**
    - Bundle only the necessary aws-sdk/client-dynamodb code by using tree shaking and only importing the necessary modules.
    - Implement lazy loading for less frequently used features.

## 2. Common Patterns and Anti-patterns

- **Design Patterns:**
    - **Repository Pattern:** Abstract DynamoDB data access logic.
    - **Data Mapper Pattern:** Transform data between application and DynamoDB formats.
    - **Single Table Design Pattern:** Minimize tables by using composite keys and attributes to represent different entity types and relationships.
    - **Composite Keys Pattern:** Combine attributes to create partition and sort keys.
- **Recommended Approaches for Common Tasks:**
    - **Creating Items:** Use the `PutItem` operation.
    - **Reading Items:** Use the `GetItem` operation for single items and `Query` for sets.
    - **Updating Items:** Use the `UpdateItem` operation for granular updates.
    - **Deleting Items:** Use the `DeleteItem` operation.
    - **Querying Data:**  Utilize indexes (GSI/LSI) to optimize query performance.  Use projection expressions to return only required attributes.
    - **Scanning Data (Avoid when possible):** If scanning is unavoidable, use `Limit` and `ExclusiveStartKey` for pagination and consider parallel scans for large datasets.
- **Anti-patterns and Code Smells:**
    - **Over-fetching Data:** Avoid retrieving unnecessary attributes.
    - **Excessive Scanning:**  Design schemas and queries to minimize scans.
    - **Hardcoding Table Names:**  Use configuration files or environment variables.
    - **Ignoring Error Handling:**  Implement robust error handling with retries and logging.
    - **Insufficient Data Validation:**  Validate data before writing to DynamoDB.
    - **Hot Partitioning:** Design partitions to distribute data evenly and prevent overload of specific partitions.
- **State Management Best Practices:**
    - Use DynamoDB to persist application state (e.g., user sessions, feature flags).
    - Implement atomic counters and conditional updates to manage concurrent access.
- **Error Handling Patterns:**
    - Use try-catch blocks to handle DynamoDB errors.
    - Implement retry mechanisms for transient errors (e.g., throttling).
    - Log errors with sufficient context for debugging.
    - Use custom exception classes to categorize DynamoDB errors.

## 3. Performance Considerations

- **Optimization Techniques:**
    - **Minimize Read Capacity Units (RCUs) and Write Capacity Units (WCUs):** Optimize queries and writes to consume fewer RCUs and WCUs. Use projection expressions to return only the attributes you need.
    - **Use Batch Operations:** Use `BatchGetItem` and `BatchWriteItem` for bulk operations.
    - **Optimize Data Sizes:**  Keep item sizes small to reduce storage costs and improve performance. If items exceed the size limit, consider storing larger attributes in S3 and referencing them in DynamoDB.
    - **Use Parallel Scans (with caution):** Use parallel scans to speed up full table scans, but be aware of the increased RCUs consumption.
    - **Optimize Index Usage:** Use indexes effectively to support query patterns.  Be mindful of the cost implications of GSI writes.
    - **Leverage DynamoDB Accelerator (DAX):** Use DAX for in-memory caching to reduce latency and RCU consumption.
- **Memory Management Considerations:**
    - Avoid loading large datasets into memory.
    - Use pagination to process data in chunks.
    - Release resources promptly after use.
- **Bundle Size Optimization Strategies:**
    - Use code splitting to reduce the initial bundle size.
    - Remove unused dependencies.
    - Optimize image sizes (if applicable).
- **Lazy Loading Strategies:**
    - Load data on demand when needed.
    - Use placeholders for content that is not immediately visible.

## 4. Security Best Practices

- **Common Vulnerabilities and Prevention:**
    - **Injection Attacks:** Prevent NoSQL injection by validating and sanitizing user inputs.
    - **Unauthorized Access:** Implement proper authentication and authorization mechanisms.
    - **Data Exposure:** Encrypt sensitive data at rest and in transit.
- **Input Validation Best Practices:**
    - Validate data types, formats, and ranges.
    - Sanitize inputs to prevent injection attacks.
    - Use input validation libraries (e.g., Joi, Validator.js).
- **Authentication and Authorization Patterns:**
    - Use AWS Identity and Access Management (IAM) roles and policies to control access to DynamoDB resources.
    - Implement fine-grained access control using conditional IAM policies.
    - Consider using AWS Cognito for user authentication and authorization.
- **Data Protection Strategies:**
    - Enable encryption at rest and in transit for DynamoDB tables.
    - Use AWS Key Management Service (KMS) to manage encryption keys.
    - Implement data masking and tokenization for sensitive data.
    - Comply with data privacy regulations (e.g., GDPR, CCPA).
- **Secure API Communication:**
    - Use HTTPS for all API communication.
    - Implement API authentication and authorization.
    - Protect against cross-site scripting (XSS) and cross-site request forgery (CSRF) attacks.

## 5. Testing Approaches

- **Unit Testing Strategies:**
    - Mock DynamoDB client using libraries like `aws-sdk-mock` or `jest-mock-extended`.
    - Test individual functions and modules in isolation.
    - Verify that functions correctly interact with the mocked DynamoDB client.
- **Integration Testing Approaches:**
    - Test the integration between different modules and components.
    - Use a local DynamoDB instance (e.g., DynamoDB Local, Docker image) for integration tests.
    - Verify that data is correctly written to and read from DynamoDB.
- **End-to-end Testing Recommendations:**
    - Test the entire application flow from the user interface to DynamoDB.
    - Use a staging environment that mirrors the production environment.
    - Verify that the application meets all functional and non-functional requirements.
- **Test Organization Best Practices:**
    - Organize tests by module or component.
    - Use descriptive names for test cases.
    - Follow the AAA (Arrange, Act, Assert) pattern.
- **Mocking and Stubbing Techniques:**
    - Use mocking libraries to create mock objects for DynamoDB client.
    - Use stubbing to replace real dependencies with controlled test values.
    - Verify that mocked methods are called with the expected parameters.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes:**
    - Forgetting to handle pagination when querying large datasets.
    - Not understanding DynamoDB capacity units (RCUs and WCUs) and throttling.
    - Using inefficient query patterns.
    - Neglecting to implement proper error handling.
    - Designing schemas that lead to hot partitions.
- **Edge Cases:**
    - Handling large item sizes exceeding DynamoDB's limits.
    - Dealing with eventual consistency when reading data.
    - Managing concurrent updates to the same item.
- **Version-Specific Issues:**
    - Be aware of breaking changes in aws-sdk/client-dynamodb versions.
    - Consult the AWS documentation and release notes for migration guides.
- **Compatibility Concerns:**
    - Ensure compatibility between aws-sdk/client-dynamodb and other libraries.
- **Debugging Strategies:**
    - Use AWS CloudWatch Logs to monitor DynamoDB operations.
    - Enable DynamoDB Streams to capture changes to DynamoDB tables.
    - Use the AWS X-Ray service for distributed tracing.

## 7. Tooling and Environment

- **Recommended Development Tools:**
    - AWS CLI
    - AWS SDK for Javascript/Typescript
    - DynamoDB Local (for local development)
    - NoSQL Workbench for DynamoDB (for data modeling).
- **Build Configuration Best Practices:**
    - Use a build tool (e.g., webpack, Parcel, esbuild) to bundle and optimize code.
    - Configure environment variables for DynamoDB table names, AWS region, and credentials.
    - Use `.env` files or AWS Systems Manager Parameter Store to manage configuration data.
- **Linting and Formatting Recommendations:**
    - Use ESLint or TSLint for linting.
    - Use Prettier for code formatting.
    - Configure linting and formatting rules to enforce consistent code style.
- **Deployment Best Practices:**
    - Use infrastructure-as-code tools (e.g., AWS CloudFormation, AWS CDK, Terraform) to provision DynamoDB resources.
    - Implement blue/green deployments or canary deployments to minimize downtime.
    - Automate deployments using CI/CD pipelines.
- **CI/CD Integration Strategies:**
    - Use CI/CD tools (e.g., Jenkins, GitLab CI, CircleCI, AWS CodePipeline) to automate the build, test, and deployment process.
    - Run unit and integration tests in the CI/CD pipeline.
    - Deploy code to staging environments for testing and validation before deploying to production.