---
description: This rule provides comprehensive guidelines for developing AI applications with LlamaIndex-JS, covering code organization, performance, security, and testing best practices. It aims to ensure robust, efficient, and secure LLM-powered applications.
globs: *.js,*.ts,*.jsx,*.tsx
---
# LlamaIndex-JS Best Practices and Coding Standards

This document outlines the best practices and coding standards for developing AI and ML applications using the LlamaIndex-JS library. Following these guidelines will help you build robust, efficient, secure, and maintainable applications.

## 1. Code Organization and Structure

### 1.1. Directory Structure

Adopt a modular directory structure to organize your LlamaIndex-JS project. Here’s a recommended structure:


my-llamaindex-app/
├── src/
│   ├── components/          # Reusable UI components (if applicable)
│   │   ├── MyComponent.tsx
│   │   └── ...
│   ├── services/           # API services and data fetching logic
│   │   ├── llamaIndexService.ts  # LlamaIndex specific functionalities
│   │   └── ...
│   ├── utils/              # Utility functions and helpers
│   │   ├── dataProcessing.ts
│   │   └── ...
│   ├── index/                # Index management
│   │   ├── documentLoaders.ts  # Custom document loaders
│   │   ├── indexBuilders.ts    # Logic for building indices
│   │   └── ...
│   ├── prompts/            # Custom prompt templates
│   │   ├── summarizationPrompt.ts
│   │   └── ...
│   ├── models/             # Data models and interfaces
│   │   ├── document.ts
│   │   └── ...
│   ├── app.ts               # Main application entry point
│   └── ...
├── tests/
│   ├── unit/
│   │   └── ...
│   ├── integration/
│   │   └── ...
│   └── ...
├── .env                   # Environment variables
├── package.json
├── tsconfig.json          # TypeScript configuration
├── README.md
└── ...


### 1.2. File Naming Conventions

*   Use descriptive and consistent file names.
*   For components, use PascalCase (e.g., `MyComponent.tsx`).
*   For utility functions and services, use camelCase (e.g., `llamaIndexService.ts`, `dataProcessing.ts`).
*   For configuration files, use kebab-case (e.g., `tsconfig.json`).

### 1.3. Module Organization

*   **Encapsulation:** Group related functions, classes, and interfaces into modules.
*   **Single Responsibility Principle:** Each module should have a clear and specific purpose.
*   **Explicit Exports:** Use explicit `export` statements to define the public API of each module.
*   **Avoid Circular Dependencies:** Be mindful of circular dependencies between modules, as they can lead to runtime errors and make code harder to understand.

### 1.4. Component Architecture (If Applicable)

*   If your LlamaIndex-JS application includes a user interface (e.g., using React), follow a component-based architecture.
*   **Presentational Components:** Focus on rendering UI elements and receiving data as props.
*   **Container Components:** Handle data fetching, state management, and logic.
*   **Component Composition:** Build complex UIs by composing smaller, reusable components.

### 1.5. Code Splitting

*   Use dynamic imports (`import()`) to split your code into smaller chunks.
*   Lazy-load components or modules that are not immediately needed.
*   This can significantly improve initial load time and reduce the overall bundle size.

typescript
// Example of lazy loading a module
async function loadMyModule() {
  const myModule = await import('./myModule');
  myModule.doSomething();
}


## 2. Common Patterns and Anti-Patterns

### 2.1. Design Patterns

*   **Retrieval-Augmented Generation (RAG):** Implement RAG to enhance LLM responses by retrieving relevant data from external sources.
*   **Factory Pattern:** Use factory functions to create instances of LlamaIndex objects, abstracting the creation logic.
*   **Strategy Pattern:** Employ different indexing or query strategies based on the specific use case.
*   **Observer Pattern:** Use this to react to changes in the underlying data or model.

### 2.2. Recommended Approaches for Common Tasks

*   **Data Loading:** Use `Document` objects and `BaseReader` classes for loading data from various sources.
*   **Indexing:** Choose appropriate index types (e.g., `VectorStoreIndex`, `SummaryIndex`) based on your data and query requirements.
*   **Querying:** Use `QueryEngine` to orchestrate complex queries and retrieve relevant information.
*   **Evaluation:** Implement evaluation metrics to measure the performance of your RAG pipeline.

### 2.3. Anti-Patterns and Code Smells

*   **Tight Coupling:** Avoid tight coupling between LlamaIndex-JS components and other parts of your application.
*   **Global State:** Minimize the use of global state, as it can make your application harder to reason about.
*   **Ignoring Errors:** Always handle errors gracefully and provide informative error messages.
*   **Over-Complicating Queries:** Keep queries simple and focused on retrieving the most relevant information.
*   **Not using `async/await`:** Use `async/await` when dealing with asynchronous operations to avoid callback hell.

### 2.4. State Management

*   Choose a state management library or pattern that fits your application's needs (e.g., React Context, Redux, Zustand).
*   Keep state minimal and derive values as needed.
*   Use immutable data structures to simplify state updates.

### 2.5. Error Handling

*   Use `try...catch` blocks to handle exceptions.
*   Provide informative error messages to the user.
*   Log errors for debugging purposes.
*   Consider using a centralized error handling mechanism.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Efficient Indexing:** Optimize indexing by selecting appropriate chunk sizes and embedding models. Experiment to find optimal values.
*   **Caching:** Cache frequently accessed data to reduce latency.
*   **Asynchronous Operations:** Use asynchronous operations to avoid blocking the main thread.
*   **Vector Store Selection:** Choose vector stores that support efficient similarity search (e.g., FAISS, Annoy, Qdrant).

### 3.2. Memory Management

*   Be mindful of memory usage, especially when dealing with large datasets.
*   Use garbage collection effectively.
*   Avoid creating unnecessary objects.
*   Use streams for processing large files.

### 3.3. Rendering Optimization (If Applicable)

*   Use virtualization techniques to render large lists efficiently.
*   Memoize components to prevent unnecessary re-renders.
*   Optimize images and other assets.

### 3.4. Bundle Size Optimization

*   Use tree shaking to remove unused code.
*   Minify and compress your code.
*   Use code splitting to load only the code that is needed for each page.
*   Analyze bundle size using tools like Webpack Bundle Analyzer.

### 3.5. Lazy Loading

*   Implement lazy loading for non-critical components or modules.
*   This can improve initial load time and reduce the overall bundle size.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Prompt Injection:** Protect against prompt injection attacks by sanitizing user input and validating LLM responses. Consider using libraries such as LLM Guard by Protect AI.
*   **Data Leakage:** Prevent data leakage by carefully controlling access to sensitive information.
*   **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks (if applicable, in UI components).
*   **API Key Exposure:** Protect API keys and other sensitive credentials by storing them securely and avoiding hardcoding them in your code.

### 4.2. Input Validation

*   Validate all user input to prevent malicious data from entering your application.
*   Use appropriate validation techniques for each type of input.
*   Sanitize input to remove potentially harmful characters.

### 4.3. Authentication and Authorization

*   Implement authentication to verify the identity of users.
*   Implement authorization to control access to resources.
*   Use strong passwords and encryption.
*   Follow the principle of least privilege.

### 4.4. Data Protection

*   Encrypt sensitive data at rest and in transit.
*   Use secure storage mechanisms.
*   Implement data loss prevention (DLP) measures.
*   Regularly back up your data.

### 4.5. Secure API Communication

*   Use HTTPS for all API communication.
*   Validate API responses to ensure data integrity.
*   Implement rate limiting to prevent abuse.
*   Use API keys or other authentication mechanisms.

## 5. Testing Approaches

### 5.1. Unit Testing

*   Write unit tests to verify the functionality of individual components and functions.
*   Use mocking and stubbing to isolate units of code.
*   Aim for high code coverage.

### 5.2. Integration Testing

*   Write integration tests to verify the interaction between different components and modules.
*   Test the integration with external services and APIs.

### 5.3. End-to-End Testing

*   Write end-to-end tests to verify the overall functionality of the application.
*   Simulate user interactions and verify the expected behavior.

### 5.4. Test Organization

*   Organize your tests into separate directories for unit, integration, and end-to-end tests.
*   Use descriptive test names.
*   Keep tests concise and focused.

### 5.5. Mocking and Stubbing

*   Use mocking and stubbing to isolate units of code during testing.
*   Use a mocking library such as Jest or Sinon.
*   Avoid over-mocking, as it can make your tests less effective.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   Incorrectly configuring the LlamaIndex-JS client.
*   Using the wrong index type for your data.
*   Not handling errors properly.
*   Ignoring performance considerations.
*   Not securing your application against vulnerabilities.

### 6.2. Edge Cases

*   Handling large documents or datasets.
*   Dealing with complex queries.
*   Handling different data formats.
*   Dealing with rate limits from external APIs.

### 6.3. Version-Specific Issues

*   Be aware of breaking changes between LlamaIndex-JS versions.
*   Consult the release notes and migration guides when upgrading.
*   Test your application thoroughly after upgrading.

### 6.4. Compatibility Concerns

*   Ensure compatibility between LlamaIndex-JS and other libraries or frameworks that you are using.
*   Test your application in different environments.

### 6.5. Debugging Strategies

*   Use a debugger to step through your code and inspect variables.
*   Log messages to the console to track the execution flow.
*   Use error monitoring tools to track errors in production.
*   Use the LlamaIndex-JS documentation and community forums to find solutions to common problems.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   Node.js
*   npm or yarn
*   TypeScript (recommended)
*   VS Code or other IDE
*   LlamaIndex Studio for visualization
*   Postman or Insomnia for testing APIs

### 7.2. Build Configuration

*   Use a build tool such as Webpack or Parcel to bundle your code.
*   Configure your build tool to optimize your code for production.
*   Use environment variables to configure your application.

### 7.3. Linting and Formatting

*   Use a linter such as ESLint to enforce coding standards.
*   Use a formatter such as Prettier to format your code consistently.
*   Configure your IDE to automatically lint and format your code.

### 7.4. Deployment

*   Choose a deployment platform that meets your needs (e.g., Vercel, Netlify, AWS).
*   Configure your deployment environment to use environment variables.
*   Set up monitoring and logging to track the performance of your application.

### 7.5. CI/CD Integration

*   Use a CI/CD platform such as GitHub Actions or Jenkins to automate your build, test, and deployment processes.
*   Configure your CI/CD pipeline to run your tests automatically.
*   Use automated deployment to deploy your application to production.

By following these best practices, you can build robust, efficient, secure, and maintainable AI applications using LlamaIndex-JS.