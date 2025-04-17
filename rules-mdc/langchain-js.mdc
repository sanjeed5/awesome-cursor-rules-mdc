---
description: Comprehensive best practices and coding standards for developing applications using LangChain.js. Focuses on code organization, performance, security, testing, and common pitfalls to ensure robust and maintainable AI-driven solutions.
globs: *.js,*.ts,*.jsx,*.tsx
---
- **Monitor and Evaluate LLM applications**: Utilize tools like LangSmith for monitoring application performance, including logging traces, analyzing latency, and evaluating LLM outputs against predefined metrics. This helps identify bottlenecks and ensures the application meets quality standards. Enable tracing by setting environment variables:
	`export LANGCHAIN_TRACING_V2="true"`
	`export LANGCHAIN_API_KEY="..."`

- **Implement Stateful Agents**: Use LangGraph to build stateful agents, crucial for applications like chatbots where remembering past interactions enhances user experience. Model interactions as a graph with nodes (states) and edges (transitions).

- **Maintain High Code Quality**: Enforce strict code quality through regular testing, ESLint, and Prettier for consistent code formatting and linting.

- **Comprehensive Documentation**: Ensure all components and their interactions are well-documented for maintainability and scalability.

- **Use LangChain Expression Language (LCEL)**: Employ LCEL for composing chains in a declarative way, supporting production deployment without code changes.

- **Explore Trade-offs in Deployment**: Choose between using external LLM providers or self-hosting open-source models based on cost, latency, and privacy considerations.

- **Secure Coding Practices**: Read up on our Security best practices to make sure you're developing safely with LangChain.

## 1. Code Organization and Structure

- **Directory Structure**: Organize code into logical modules based on functionality (e.g., `chains`, `agents`, `tools`, `memory`).
  
  src/
  ├── chains/
  │   ├── conversationalChain.ts
  │   └── summarizationChain.ts
  ├── agents/
  │   ├── agent.ts
  │   └── agentExecutor.ts
  ├── tools/
  │   ├── searchTool.ts
  │   └── calculatorTool.ts
  ├── memory/
  │   ├── bufferMemory.ts
  │   └── conversationBufferMemory.ts
  ├── utils/
  │   ├── api.ts
  │   └── helpers.ts
  ├── index.ts
  └── types.ts
  

- **File Naming Conventions**: Use descriptive names, typically in `camelCase` for variables and functions, and `PascalCase` for classes and interfaces.  Consider prefixes or suffixes to denote the type of module e.g., `*_chain.ts` or `*_agent.ts`

- **Module Organization**: Group related functionalities into modules with clear interfaces. Use `index.ts` files to export module members for cleaner imports.
  typescript
  // chains/index.ts
  export * from './conversationalChain';
  export * from './summarizationChain';
  
  // Usage:
  import { ConversationalChain, SummarizationChain } from '@/chains';
  

- **Component Architecture**: Design modular components for reusability. Implement interfaces to define contracts between components.
  typescript
  // Define an interface for a Tool
  interface ToolInterface {
      name: string;
      description: string;
      execute(input: string): Promise<string>;
  }

  // Implement the interface in a specific Tool
  class SearchTool implements ToolInterface {
      name = 'search';
      description = 'Useful for searching the internet.';
      async execute(input: string): Promise<string> {
          // Implementation
      }
  }
  

- **Code Splitting**: Implement code splitting using dynamic imports to reduce initial load time, especially for large applications.
  typescript
  async function loadLargeModule() {
      const largeModule = await import('./largeModule');
      largeModule.initialize();
  }
  

## 2. Common Patterns and Anti-patterns

- **Design Patterns**: Use the Factory pattern for creating different types of chains or agents, and the Strategy pattern for choosing different LLMs.
  typescript
  // Factory Pattern for creating chains
  class ChainFactory {
      static createChain(type: 'conversational' | 'summarization', llm: LLM) {
          if (type === 'conversational') {
              return new ConversationalChain(llm);
          } else if (type === 'summarization') {
              return new SummarizationChain(llm);
          } else {
              throw new Error('Invalid chain type');
          }
      }
  }

  const chain = ChainFactory.createChain('conversational', new OpenAIChat());
  

- **Recommended Approaches**: Leverage LangChain's built-in modules and chains when possible, and customize them as needed. Prioritize asynchronous operations to prevent blocking the main thread.

- **Anti-patterns**: Avoid deeply nested callbacks, which can lead to callback hell. Use `async/await` and promises for cleaner asynchronous code.

- **State Management**: For simple applications, manage state with React's `useState` or similar. For complex applications, consider state management libraries like Zustand or Redux.

- **Error Handling**: Implement robust error handling with `try/catch` blocks and global error handlers. Log errors with context for debugging.
  typescript
  async function processData() {
      try {
          const result = await fetchData();
          // Process result
      } catch (error) {
          console.error('Error processing data:', error);
          // Handle error (e.g., display error message to user)
      }
  }
  

## 3. Performance Considerations

- **Optimization Techniques**: Use caching to store and reuse LLM responses. Optimize prompts to reduce token usage. Debounce computationally expensive operations.

- **Memory Management**: Be mindful of memory leaks, especially when using streams or subscriptions. Properly clean up resources when components unmount.

- **Bundle Size Optimization**: Use tools like Webpack Bundle Analyzer to identify large dependencies. Use tree shaking to remove unused code.

- **Lazy Loading**: Implement lazy loading for components and modules that are not immediately needed to improve initial load time.

## 4. Security Best Practices

- **Vulnerabilities**: Prevent prompt injection attacks by carefully validating user inputs and using sandboxed execution environments.

- **Input Validation**: Sanitize user inputs to prevent malicious code execution. Limit the length of inputs to prevent denial-of-service attacks.

- **Authentication/Authorization**: Use secure authentication and authorization mechanisms to protect sensitive data and prevent unauthorized access.

- **Data Protection**: Encrypt sensitive data at rest and in transit. Comply with data privacy regulations like GDPR and CCPA.

- **Secure API Communication**: Use HTTPS for all API communication. Validate API responses to prevent data corruption.

## 5. Testing Approaches

- **Unit Testing**: Test individual components in isolation using Jest or Mocha. Mock dependencies to control test environments.

- **Integration Testing**: Test interactions between components to ensure they work together correctly. Use in-memory databases or mock APIs for integration tests.

- **End-to-end Testing**: Test the entire application flow using tools like Cypress or Puppeteer. Simulate user interactions to verify functionality.

- **Test Organization**: Organize tests into separate directories based on component or module. Use clear and descriptive test names.

- **Mocking/Stubbing**: Use mocking libraries like Jest's `jest.mock()` or Sinon.js to replace dependencies with controlled test doubles.

## 6. Common Pitfalls and Gotchas

- **Frequent Mistakes**: Incorrectly configuring API keys, not handling errors properly, and overlooking prompt injection vulnerabilities.

- **Edge Cases**: Handling unexpected input formats, dealing with rate limits from LLM providers, and managing long-running operations.

- **Version-Specific Issues**: Check release notes for breaking changes when upgrading LangChain.js versions.

- **Compatibility Concerns**: Ensure compatibility between LangChain.js and other libraries, especially those related to data processing or UI frameworks.

- **Debugging Strategies**: Use console logging, debuggers, and monitoring tools like LangSmith to diagnose issues.

## 7. Tooling and Environment

- **Recommended Tools**: VS Code with TypeScript support, ESLint, Prettier, and Jest.

- **Build Configuration**: Use Webpack, Parcel, or Rollup for bundling and optimization. Configure TypeScript compiler options for strict type checking.

- **Linting/Formatting**: Enforce consistent code style with ESLint and Prettier. Use linting rules to catch potential errors and enforce best practices.

- **Deployment Best Practices**: Use serverless functions or containerization for deployment. Implement monitoring and alerting to detect issues in production.

- **CI/CD Integration**: Automate testing, linting, and deployment with CI/CD pipelines using tools like GitHub Actions or Jenkins.