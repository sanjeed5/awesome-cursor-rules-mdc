---
description: This rule outlines best practices and coding standards for developing with LlamaIndex, covering code organization, performance, security, testing, and common pitfalls. It aims to ensure maintainable, efficient, and secure LlamaIndex applications.
globs: *.py
---
# LlamaIndex Best Practices and Coding Standards

This document provides comprehensive guidance on developing high-quality applications using LlamaIndex. It covers various aspects of development, including code organization, performance optimization, security considerations, and testing strategies.

## 1. Code Organization and Structure

*   **Directory Structure Best Practices:**
    *   `data/`: Store data sources (e.g., documents, PDFs) used by LlamaIndex.
    *   `indices/`: Contains index definitions and configurations.
    *   `queries/`: Defines query engines and query logic.
    *   `models/`: Place custom LLM or embedding model configurations.
    *   `utils/`: Utility functions and helper classes.
    *   `tests/`: Unit, integration, and end-to-end tests.
    *   `scripts/`: Scripts for data ingestion, index building, or other automation tasks.
*   **File Naming Conventions:**
    *   Data loaders: `*_loader.py` (e.g., `pdf_loader.py`)
    *   Index definitions: `*_index.py` (e.g., `vector_index.py`)
    *   Query engines: `*_query_engine.py` (e.g., `knowledge_graph_query_engine.py`)
    *   Models: `*_model.py` (e.g., `custom_llm_model.py`)
    *   Utilities: `*_utils.py` (e.g., `text_processing_utils.py`)
    *   Tests: `test_*.py` (e.g., `test_vector_index.py`)
*   **Module Organization:**
    *   Group related functionalities into modules (e.g., `data_ingestion`, `indexing`, `querying`).
    *   Use clear and descriptive module names.
    *   Minimize dependencies between modules to improve maintainability.
*   **Component Architecture:**
    *   **Data Connectors:** Abstract data loading logic into reusable connectors.
    *   **Index Structures:** Use appropriate index structures (e.g., `VectorStoreIndex`, `KnowledgeGraphIndex`) based on data characteristics and query requirements.
    *   **Query Engines:** Decouple query logic from index structures.
    *   **LLM Abstraction:** Abstract LLM calls using interfaces for flexibility and testability.
*   **Code Splitting:**
    *   Break down large functions into smaller, well-defined functions.
    *   Use classes to encapsulate related data and behavior.
    *   Extract reusable code into separate modules or packages.

## 2. Common Patterns and Anti-patterns

*   **Design Patterns:**
    *   **Factory Pattern:** For creating different types of indexes or query engines.
    *   **Strategy Pattern:** For choosing different retrieval or ranking algorithms.
    *   **Decorator Pattern:** For adding pre-processing or post-processing steps to queries.
*   **Recommended Approaches:**
    *   **Data Ingestion:** Use `SimpleDirectoryReader` or custom data connectors to load data.
    *   **Indexing:** Choose the appropriate index type based on your data and query needs. Consider `VectorStoreIndex` for semantic search, `KnowledgeGraphIndex` for knowledge graph-based queries, and `ComposableGraph` for combining multiple indexes.
    *   **Querying:** Use `as_query_engine()` to create a query engine from an index. Customize the query engine with different retrieval and response synthesis modules.
    *   **Evaluation:** Use LlamaIndex's evaluation modules to measure the performance of your LLM application (e.g., retrieval and LLM response quality).
*   **Anti-patterns and Code Smells:**
    *   **Tight Coupling:** Avoid tight coupling between components. Use interfaces and dependency injection to promote loose coupling.
    *   **God Classes:** Avoid creating large classes that do too much. Break them down into smaller, more focused classes.
    *   **Code Duplication:** Avoid duplicating code. Extract common code into reusable functions or classes.
    *   **Ignoring Errors:** Don't ignore errors. Handle them gracefully or raise exceptions.
*   **State Management:**
    *   Use LlamaIndex's `StorageContext` to persist indexes to disk.
    *   Consider using a database to store application state.
*   **Error Handling:**
    *   Use `try-except` blocks to handle exceptions.
    *   Log errors for debugging purposes.
    *   Provide informative error messages to the user.

## 3. Performance Considerations

*   **Optimization Techniques:**
    *   **Indexing:** Optimize index construction by using appropriate chunk sizes and overlap.
    *   **Querying:** Optimize query performance by using appropriate retrieval and ranking algorithms.
    *   **Caching:** Cache query results to improve performance.
    *   **Parallelization:** Parallelize data loading and indexing tasks.
*   **Memory Management:**
    *   Use generators to process large datasets in chunks.
    *   Release memory when it is no longer needed.
*   **Bundle Size Optimization:** (Not directly applicable to LlamaIndex as it is a backend library, but relevant if building a web UI on top)
    *   Remove unused code.
    *   Use code splitting to load only the code that is needed.
*   **Lazy Loading:**
    *   Load data and models only when they are needed.
    *   Use lazy initialization to defer the creation of objects until they are first used.

## 4. Security Best Practices

*   **Common Vulnerabilities:**
    *   **Prompt Injection:** Prevent prompt injection attacks by carefully sanitizing user input and using appropriate prompt engineering techniques.
    *   **Data Leaks:** Protect sensitive data by using appropriate access control and encryption.
    *   **API Key Exposure:** Avoid exposing API keys in your code. Use environment variables or a secure configuration management system to store API keys.
*   **Input Validation:**
    *   Validate all user input to prevent injection attacks.
    *   Sanitize input to remove potentially harmful characters.
*   **Authentication and Authorization:**
    *   Implement authentication and authorization to control access to your application.
    *   Use strong passwords and multi-factor authentication.
*   **Data Protection:**
    *   Encrypt sensitive data at rest and in transit.
    *   Use appropriate access control to protect data.
*   **Secure API Communication:**
    *   Use HTTPS to encrypt communication between your application and the LlamaIndex API.
    *   Validate the server certificate to prevent man-in-the-middle attacks.

## 5. Testing Approaches

*   **Unit Testing:**
    *   Write unit tests for all core components, including data connectors, index structures, and query engines.
    *   Use mocking and stubbing to isolate components during testing.
*   **Integration Testing:**
    *   Write integration tests to verify that different components work together correctly.
    *   Test the integration between LlamaIndex and other libraries or frameworks.
*   **End-to-end Testing:**
    *   Write end-to-end tests to verify that the entire application works as expected.
    *   Test the application with real data and user scenarios.
*   **Test Organization:**
    *   Organize tests into separate directories for unit, integration, and end-to-end tests.
    *   Use clear and descriptive test names.
*   **Mocking and Stubbing:**
    *   Use mocking and stubbing to isolate components during testing.
    *   Use a mocking framework such as `unittest.mock` or `pytest-mock`.

## 6. Common Pitfalls and Gotchas

*   **Frequent Mistakes:**
    *   Using the wrong index type for the data.
    *   Not optimizing query performance.
    *   Not handling errors gracefully.
    *   Exposing API keys.
    *   Not validating user input.
*   **Edge Cases:**
    *   Handling large documents.
    *   Handling noisy or incomplete data.
    *   Handling complex queries.
*   **Version-Specific Issues:**
    *   Be aware of breaking changes in LlamaIndex releases.
    *   Refer to the LlamaIndex documentation for version-specific information.
*   **Compatibility Concerns:**
    *   Ensure that LlamaIndex is compatible with the other libraries and frameworks that you are using.
    *   Test your application thoroughly to identify any compatibility issues.
*   **Debugging Strategies:**
    *   Use logging to track the execution of your application.
    *   Use a debugger to step through your code and inspect variables.
    *   Use LlamaIndex's debugging tools to diagnose issues.

## 7. Tooling and Environment

*   **Recommended Development Tools:**
    *   **IDE:** VS Code, PyCharm
    *   **Package Manager:** Poetry, pip
    *   **Testing Framework:** pytest
    *   **Linting and Formatting:** flake8, black
*   **Build Configuration:**
    *   Use a build system such as `poetry` to manage dependencies.
    *   Create a `requirements.txt` file to list dependencies.
*   **Linting and Formatting:**
    *   Use a linter such as `flake8` to enforce code style.
    *   Use a formatter such as `black` to automatically format code.
*   **Deployment Best Practices:**
    *   Use a containerization technology such as Docker to package your application.
    *   Use a cloud platform such as AWS, Azure, or GCP to deploy your application.
*   **CI/CD Integration:**
    *   Use a CI/CD system such as GitHub Actions or Jenkins to automate the build, test, and deployment process.