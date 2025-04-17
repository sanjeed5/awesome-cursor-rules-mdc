---
description: This rule file provides comprehensive best practices for developing with LangGraph, covering code organization, performance, security, testing, and common pitfalls. It offers actionable guidance for developers to build robust and maintainable LangGraph applications.
globs: *.py
---
# LangGraph Best Practices and Coding Standards

This document outlines best practices and coding standards for developing with LangGraph. It aims to provide clear, actionable guidance for developers to build robust, maintainable, and scalable LangGraph applications. It covers various aspects, including code organization, common patterns, performance considerations, security, testing, and common pitfalls.

## Library Information:

- Name: langgraph
- Tags: ai, ml, llm, python, agent-framework, workflow

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices


my_langgraph_project/
├── data/                      # Datasets, knowledge bases, or other data files.
├── src/                       # Source code.
│   ├── components/             # Reusable components (e.g., custom nodes, tools).
│   │   ├── __init__.py
│   │   ├── retrieval.py        # Retrieval-related nodes
│   │   ├── tool_selector.py    # Logic for selecting which tool to use
│   │   └── ...
│   ├── graphs/                 # Graph definitions.
│   │   ├── __init__.py
│   │   ├── customer_support.py  # Example: Customer support graph.
│   │   ├── rag_pipeline.py     # Example: RAG pipeline graph.
│   │   └── ...
│   ├── utils/                  # Utility functions and helpers.
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration loading
│   │   ├── logging.py          # Logging setup
│   │   └── ...
│   ├── schemas/                # Data schemas and type definitions.
│   │   ├── __init__.py
│   │   ├── agent_state.py      # Definition of agent state
│   │   └── ...
│   ├── main.py                 # Entry point of the application.
│   └── ...
├── tests/                     # Unit and integration tests.
│   ├── __init__.py
│   ├── components/             # Tests for custom components.
│   ├── graphs/                 # Tests for graph definitions.
│   ├── utils/                  # Tests for utility functions.
│   └── ...
├── .env                       # Environment variables.
├── requirements.txt           # Project dependencies.
├── pyproject.toml            # Project metadata and build settings
└── README.md                  # Project documentation.


### 1.2. File Naming Conventions

-   Python files: `snake_case.py` (e.g., `customer_support.py`, `retrieval_node.py`).
-   Class names: `PascalCase` (e.g., `CustomerSupportGraph`, `RetrievalNode`).
-   Variables and functions: `snake_case` (e.g., `user_query`, `process_message`).
-   Configuration files: `config.yaml` or `config.json`

### 1.3. Module Organization Best Practices

-   Group related functionalities into modules (e.g., `components`, `graphs`, `utils`).
-   Use `__init__.py` files to make directories packages.
-   Keep modules focused and avoid overly large files.
-   Use relative imports within modules to avoid naming conflicts.

### 1.4. Component Architecture Recommendations

-   Design reusable components for common tasks (e.g., data retrieval, text summarization, tool selection).
-   Create abstract base classes or interfaces for components to promote code reuse and modularity.
-   Use dependency injection to configure components and their dependencies.
-   Adhere to the Single Responsibility Principle (SRP) when designing components.

### 1.5. Code Splitting Strategies

-   Split large graph definitions into smaller, more manageable files.
-   Use lazy loading for components that are not immediately needed.
-   Consider using a module bundler (e.g., esbuild via a plugin) to optimize bundle size for deployment.
-   Break the system down into microservices if warranted by scale and complexity of the overall system. Communicate between microservices using REST or message queues.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

-   **State Management Pattern**: Encapsulate the agent state in a dedicated class or data structure to ensure consistency and maintainability. Use LangGraph's `StateGraph` to clearly define the state transitions.
-   **Node Pattern**: Define reusable nodes for common tasks such as information retrieval, tool selection, and response generation.
-   **Conditional Edge Pattern**: Use conditional edges to implement branching logic based on the agent state or external factors. This makes the graph more dynamic and responsive.
-   **Retry Pattern:** Implement retry logic within nodes or edges to handle transient errors or API rate limits.  Use exponential backoff to avoid overwhelming failing services.
-   **Orchestration Pattern**: Use LangGraph as the orchestrator for complex agentic workflows, delegating specific tasks to specialized components or services.

### 2.2. Recommended Approaches for Common Tasks

-   **Information Retrieval**: Use LangChain's retrieval chain or custom nodes to fetch relevant information from external sources.
-   **Tool Selection**: Implement a tool selection node that dynamically chooses the appropriate tool based on the user query and agent state.
-   **Response Generation**: Use LangChain's LLMChain or custom nodes to generate responses based on the retrieved information and agent state.
-   **Error Handling:** Implement robust error handling within nodes and edges to gracefully handle exceptions and prevent application crashes. Log all errors and implement monitoring to quickly detect and resolve issues.

### 2.3. Anti-patterns and Code Smells

-   **Monolithic Graphs**: Avoid creating overly complex graphs with too many nodes and edges. Break them down into smaller, more manageable subgraphs.
-   **Hardcoded Values**: Avoid hardcoding values directly into the graph definition. Use configuration files or environment variables to manage configurable parameters.
-   **Ignoring Errors**: Always handle exceptions and log errors appropriately. Ignoring errors can lead to unexpected behavior and difficult-to-debug issues.
-   **Over-Reliance on Global State**: Minimize the use of global state to avoid unintended side effects and make the application more testable.
-   **Lack of Testing**: Thoroughly test all components and graph definitions to ensure they function correctly and handle edge cases.
-   **Infinite Loops:** Ensure the conditional edges within the graph are well-defined to avoid infinite loops.

### 2.4. State Management Best Practices

-   Define a clear and concise agent state schema.
-   Use immutable data structures for the agent state to avoid accidental modifications.
-   Persist the agent state to a database or other storage medium to support long-running conversations or task executions.  Consider using vector databases for efficient retrieval.
-   Implement versioning for the agent state schema to support schema migrations.
-   Use LangGraph's checkpointing feature to save and restore the agent state.

### 2.5. Error Handling Patterns

-   Use try-except blocks to catch exceptions within nodes and edges.
-   Log all errors and warnings with relevant context information.
-   Implement retry logic for transient errors or API rate limits.
-   Use fallback mechanisms to gracefully handle unrecoverable errors.
-   Centralize error handling logic in a dedicated module or class.
-   Implement circuit breaker pattern to prevent cascading failures.

## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Caching**: Implement caching for frequently accessed data or LLM responses.
-   **Batching**: Batch multiple requests to external APIs to reduce latency.
-   **Asynchronous Operations**: Use asynchronous operations to perform non-blocking I/O and improve responsiveness.
-   **Parallel Processing**: Use multi-threading or multi-processing to parallelize computationally intensive tasks.
-   **Graph Optimization**: Optimize the graph structure to minimize the number of nodes and edges.
-   **Prompt Optimization**: Carefully design prompts to reduce the number of tokens and improve LLM performance.
-   **Reduce LLM calls**: Cache LLM responses when possible. Fine-tune smaller models for specific tasks to reduce latency and cost.

### 3.2. Memory Management Considerations

-   Monitor memory usage to detect memory leaks or excessive memory consumption.
-   Use garbage collection to reclaim unused memory.
-   Avoid storing large objects in the agent state.
-   Use streaming or lazy loading for large data sets.

### 3.3. (Not applicable, as LangGraph doesn't directly handle rendering)

### 3.4. Bundle Size Optimization

-   Use a module bundler (e.g., esbuild) to optimize bundle size.
-   Remove unused code and dependencies.
-   Use code splitting to load only the necessary code for each route or component.
-   Compress the bundle using gzip or Brotli.

### 3.5. Lazy Loading Strategies

-   Use lazy loading for components that are not immediately needed.
-   Load large data sets or models on demand.
-   Implement code splitting to load only the necessary code for each graph or component.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and Prevention

-   **Prompt Injection**: Prevent prompt injection by carefully validating user inputs and sanitizing prompts.
-   **Data Leaks**: Protect sensitive data by encrypting it at rest and in transit.
-   **Unauthorized Access**: Implement strong authentication and authorization mechanisms to control access to the application and its data.
-   **Denial of Service (DoS)**: Implement rate limiting and request filtering to prevent DoS attacks.
-   **Code Injection**: Avoid executing arbitrary code based on user inputs to prevent code injection vulnerabilities.
-   **API Key Exposure**: Store API keys securely using environment variables or a secrets management system and avoid committing them to version control.

### 4.2. Input Validation

-   Validate all user inputs to prevent prompt injection and other vulnerabilities.
-   Use regular expressions or other validation techniques to ensure that inputs conform to the expected format.
-   Sanitize inputs to remove potentially harmful characters or code.
-   Enforce input length limits to prevent buffer overflows.

### 4.3. Authentication and Authorization

-   Use strong authentication mechanisms (e.g., multi-factor authentication) to verify user identities.
-   Implement role-based access control (RBAC) to restrict access to sensitive data and functionality.
-   Use secure session management to protect user sessions from hijacking.
-   Store passwords securely using hashing and salting.

### 4.4. Data Protection

-   Encrypt sensitive data at rest and in transit.
-   Use secure protocols (e.g., HTTPS) for all API communication.
-   Implement data masking to protect sensitive data from unauthorized access.
-   Regularly back up data to prevent data loss.
-   Comply with relevant data privacy regulations (e.g., GDPR, CCPA).

### 4.5. Secure API Communication

-   Use HTTPS for all API communication.
-   Implement API authentication and authorization.
-   Validate API requests and responses.
-   Use rate limiting to prevent API abuse.
-   Monitor API traffic for suspicious activity.

## 5. Testing Approaches

### 5.1. Unit Testing

-   Write unit tests for all components and utility functions.
-   Use mocking and stubbing to isolate components during testing.
-   Test edge cases and error conditions.
-   Aim for high test coverage.

### 5.2. Integration Testing

-   Write integration tests to verify the interactions between different components.
-   Test the integration with external APIs and services.
-   Use a test environment that closely resembles the production environment.

### 5.3. End-to-End Testing

-   Write end-to-end tests to verify the entire application flow.
-   Use a testing framework such as Playwright or Selenium to automate end-to-end tests.
-   Test the application from the user's perspective.

### 5.4. Test Organization

-   Organize tests into separate directories for unit tests, integration tests, and end-to-end tests.
-   Use descriptive names for test files and test functions.
-   Follow a consistent naming convention for test files and test functions.
-   Use test suites to group related tests.

### 5.5. Mocking and Stubbing

-   Use mocking to replace external dependencies with mock objects.
-   Use stubbing to replace complex components with simplified versions.
-   Use a mocking framework such as pytest-mock to simplify mocking and stubbing.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

-   **Incorrect State Management**: Failing to properly manage the agent state can lead to inconsistent behavior and incorrect results.
-   **Ignoring Edge Cases**: Neglecting to handle edge cases can cause unexpected errors and application crashes.
-   **Over-Engineering**: Over-complicating the graph definition can make it difficult to understand and maintain.
-   **Insufficient Testing**: Lack of thorough testing can lead to undetected bugs and application failures.
-   **Not Handling Asynchronous Operations Correctly:** LangGraph, and LLMs generally, use async operations, and failing to await these operations will cause unpredictable results.

### 6.2. Edge Cases

-   **Empty User Inputs**: Handle cases where the user provides empty or invalid inputs.
-   **API Rate Limits**: Implement retry logic and rate limiting to handle API rate limits.
-   **Unexpected API Responses**: Handle cases where external APIs return unexpected responses.
-   **Large Data Sets**: Use streaming or lazy loading to handle large data sets.

### 6.3. Version-Specific Issues

-   Be aware of compatibility issues between different versions of LangGraph and LangChain.
-   Consult the documentation and release notes for any version-specific issues.
-   Pin dependencies to specific versions to avoid unexpected behavior.

### 6.4. Compatibility Concerns

-   Ensure compatibility between LangGraph and other technologies used in the application.
-   Test the integration with external APIs and services.
-   Use a consistent set of libraries and dependencies.

### 6.5. Debugging Strategies

-   Use logging to track the execution flow and identify errors.
-   Use a debugger to step through the code and inspect variables.
-   Use a testing framework to write unit tests and integration tests.
-   Use monitoring tools to track performance and identify bottlenecks.
-   Visualize the graph structure to understand the flow of execution.

## 7. Tooling and Environment

### 7.1. Recommended Tools

-   **IDE**: PyCharm, Visual Studio Code with Python extension.
-   **Virtual Environment Manager**: venv, conda.
-   **Testing Framework**: pytest.
-   **Mocking Framework**: pytest-mock.
-   **Linting and Formatting**: pylint, black.
-   **Module Bundler**: esbuild via a plugin.
-   **CI/CD**: GitHub Actions, GitLab CI.
-   **Secrets Management**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault.
-   **Monitoring**: LangSmith, Prometheus, Grafana.

### 7.2. Build Configuration

-   Use a build system such as `poetry` or `pip` to manage dependencies.
-   Use a configuration file such as `pyproject.toml` or `setup.py` to define project metadata and build settings.

### 7.3. Linting and Formatting

-   Use a linter such as `pylint` or `flake8` to enforce code style and identify potential errors.
-   Use a code formatter such as `black` or `autopep8` to automatically format the code.
-   Configure the linter and formatter to use a consistent set of rules and settings.

### 7.4. Deployment

-   Containerize the application using Docker.
-   Deploy the application to a cloud platform such as AWS, Azure, or Google Cloud.
-   Use a deployment tool such as Terraform or Ansible to automate the deployment process.
-   Implement a blue-green deployment strategy to minimize downtime.

### 7.5. CI/CD

-   Use a CI/CD tool such as GitHub Actions or GitLab CI to automate the testing, building, and deployment processes.
-   Configure the CI/CD pipeline to run tests, linters, and formatters.
-   Use a code review process to ensure code quality and security.

## Conclusion

By following these best practices and coding standards, developers can build robust, maintainable, and scalable LangGraph applications. This will also help with collaboration amongst team members working in the same codebase.