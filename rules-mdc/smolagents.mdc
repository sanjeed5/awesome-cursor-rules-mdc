---
description: This rule provides comprehensive best practices for developing with the smolagents library, covering code organization, performance, security, testing, and common pitfalls. It aims to guide developers in building robust, maintainable, and efficient AI agent applications.
globs: *.py
---
# smolagents Best Practices and Coding Standards

This document outlines the best practices and coding standards for developing with the smolagents library. Following these guidelines will help you create robust, maintainable, and efficient AI agent applications.

## Library Information:
- Name: smolagents
- Tags: ai, ml, llm, python, agent-framework, lightweight

## Core Principles
- **Simplicity:** Embrace smolagents' focus on minimal code and clear abstractions.
- **Documentation:** Thoroughly document your code, as AI models lack memory and context.
- **Security:** Prioritize secure coding practices to prevent vulnerabilities.
- **Code Review:** Regularly review code to catch design errors and maintain quality.
- **Code-First Approach:** Utilize smolagents' code-first approach, where agents write their actions in Python.

## 1. Code Organization and Structure

### 1.1 Directory Structure Best Practices


project_root/
├── agents/
│   ├── __init__.py
│   ├── my_agent.py  # Agent definitions
│   └── ...
├── tools/
│   ├── __init__.py
│   ├── web_search_tool.py # Custom tool definitions
│   └── ...
├── models/
│   ├── __init__.py
│   ├── llm_model.py  # Model configurations
│   └── ...
├── config/
│   ├── config.py      # Configuration settings
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── test_agents.py # Unit and integration tests
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── helper_functions.py # Helper functions
│   └── ...
├── main.py        # Entry point for the application
├── requirements.txt # Project dependencies
├── README.md      # Project documentation
└── .env           # Environment variables


-   **agents/:**  Contains the definitions for your AI agents.
-   **tools/:**   Holds custom tools that your agents can use.
-   **models/:**  Defines configurations for LLMs (e.g., Hugging Face models).
-   **config/:**  Stores configuration settings for your application.
-   **tests/:**   Includes unit, integration, and end-to-end tests.
-   **utils/:**   Contains helper functions and utilities.
-   **main.py:** The entry point for your application.
-   **.env:** Store sensitive information and configuration variables.

### 1.2 File Naming Conventions

-   Use descriptive and consistent names for files and modules.
-   Agent files: `my_agent.py`, `task_specific_agent.py`
-   Tool files: `web_search_tool.py`, `data_analysis_tool.py`
-   Model files: `llm_model.py`, `embedding_model.py`
-   Configuration files: `config.py`, `api_keys.py`
-   Test files: `test_agents.py`, `test_tools.py`

### 1.3 Module Organization

-   Group related functionalities into modules.
-   Use `__init__.py` files to define packages.
-   Keep modules focused on a specific task or domain.

### 1.4 Component Architecture

-   Design your application with reusable components.
-   Agents, tools, and models should be modular and easily swappable.
-   Use interfaces to define contracts between components.

### 1.5 Code Splitting Strategies

-   Break down large agent implementations into smaller, manageable functions and classes.
-   Consider splitting code based on different stages of the agent's workflow (e.g., planning, execution, evaluation).
-   Use separate modules for reusable logic.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

-   **Tool Pattern:** Encapsulate external functionalities (e.g., web search, data analysis) into reusable tools. Example: 
    python
    from smolagents import tool

    @tool
    def search_wikipedia(query: str) -> str:
        """Searches Wikipedia for the given query."""
        # Implementation
        pass
    
-   **Chain of Responsibility:** Implement agent workflows as a chain of tasks or sub-agents.
-   **Strategy Pattern:** Allow agents to switch between different strategies (e.g., different reasoning approaches) at runtime.

### 2.2 Recommended Approaches

-   **Secure Code Execution:** Use sandboxed environments (e.g., E2B) to execute agent code safely.
-   **Hub Integration:** Leverage the Hugging Face Hub for tool sharing and discovery.
-   **Model Agnosticism:** Design your agents to work with different LLMs.

### 2.3 Anti-patterns and Code Smells

-   **Hardcoding API Keys:**  Store API keys in environment variables or configuration files, not directly in the code.
-   **Ignoring Errors:** Implement proper error handling and logging.
-   **Overly Complex Agents:** Keep agent logic simple and focused.
-   **Lack of Documentation:** Always document your code, especially agent logic.

### 2.4 State Management

-   Use classes to encapsulate agent state.
-   Consider using a state management library for more complex applications.
-   Avoid global state.

### 2.5 Error Handling

-   Use `try-except` blocks to handle exceptions.
-   Log errors with meaningful messages.
-   Implement retry mechanisms for transient errors.
-   Raise custom exceptions for specific error conditions.

## 3. Performance Considerations

### 3.1 Optimization Techniques

-   **Caching:** Cache LLM responses to avoid redundant API calls.
-   **Asynchronous Operations:** Use asynchronous operations for I/O-bound tasks.
-   **Efficient Data Structures:** Choose appropriate data structures for your use case.
-   **Prompt Optimization:** Craft prompts carefully to reduce LLM processing time.

### 3.2 Memory Management

-   Be mindful of memory usage when processing large datasets.
-   Use generators to process data in chunks.
-   Delete unnecessary objects to free up memory.

### 3.3 Rendering Optimization (If Applicable)

-   If your agent interacts with a user interface, optimize rendering performance by minimizing DOM updates and using efficient rendering techniques.

### 3.4 Bundle Size Optimization (If Applicable)

-   If you are deploying your agent as a web application, optimize bundle size by removing unused code and using code splitting.

### 3.5 Lazy Loading

-   Load modules and data only when they are needed to improve startup time.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

-   **Prompt Injection:** Protect against malicious prompts that can manipulate the agent's behavior.
-   **Code Injection:** Prevent agents from executing arbitrary code.
-   **Data Leakage:** Ensure that sensitive data is not exposed.

### 4.2 Input Validation

-   Validate all inputs to your agents to prevent injection attacks.
-   Sanitize inputs to remove potentially harmful characters.

### 4.3 Authentication and Authorization

-   Implement authentication and authorization to control access to your agents.
-   Use secure credentials management practices.

### 4.4 Data Protection

-   Encrypt sensitive data at rest and in transit.
-   Use secure storage solutions for sensitive data.

### 4.5 Secure API Communication

-   Use HTTPS for all API communication.
-   Validate API responses to prevent data tampering.

## 5. Testing Approaches

### 5.1 Unit Testing

-   Write unit tests for individual components (agents, tools, models).
-   Use mocking to isolate components during testing.

### 5.2 Integration Testing

-   Test the interaction between different components.
-   Verify that the agent workflow works as expected.

### 5.3 End-to-End Testing

-   Test the entire application from end to end.
-   Simulate user interactions to verify functionality.

### 5.4 Test Organization

-   Organize tests into separate modules based on functionality.
-   Use clear and descriptive test names.

### 5.5 Mocking and Stubbing

-   Use mocking to replace external dependencies with controlled substitutes.
-   Use stubbing to provide predefined responses for specific inputs.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

-   **Ignoring Documentation:**  Failing to read the smolagents documentation thoroughly.
-   **Overcomplicating Agents:** Creating agents that are too complex and difficult to maintain.
-   **Not Using Sandboxed Environments:** Running agent code without proper security measures.
-   **Failing to Update Documentation:**  Allowing code documentation to become outdated.

### 6.2 Edge Cases

-   Handle edge cases and unexpected inputs gracefully.
-   Test your agents with a variety of inputs to identify potential issues.

### 6.3 Version-Specific Issues

-   Be aware of version-specific issues and compatibility concerns when upgrading smolagents.
-   Consult the release notes for information on breaking changes.

### 6.4 Compatibility Concerns

-   Ensure that smolagents is compatible with other technologies you are using.
-   Test your application with different versions of dependencies.

### 6.5 Debugging Strategies

-   Use logging to track the execution of your agents.
-   Use a debugger to step through code and inspect variables.
-   Use print statements to debug simple issues.

## 7. Tooling and Environment

### 7.1 Recommended Tools

-   **VS Code:**  A popular code editor with excellent Python support and extensions for AI development.
-   **PyCharm:**  A dedicated Python IDE with advanced features for debugging and testing.
-   **Docker:**  A containerization platform for deploying your agents in isolated environments.
-   **E2B:** A secure sandboxed environment for executing agent code.

### 7.2 Build Configuration

-   Use a build system like `poetry` or `pipenv` to manage dependencies.
-   Define build scripts for common tasks like testing and linting.

### 7.3 Linting and Formatting

-   Use a linter like `pylint` or `flake8` to enforce code style.
-   Use a formatter like `black` to automatically format your code.

### 7.4 Deployment

-   Deploy your agents to a cloud platform like AWS, Azure, or Google Cloud.
-   Use containerization to ensure consistent deployment across different environments.

### 7.5 CI/CD

-   Integrate your project with a CI/CD pipeline to automate testing and deployment.
-   Use a CI/CD tool like Jenkins, GitLab CI, or GitHub Actions.

## Additional Recommendations

-  **Prompt Engineering:**
    - Be extremely careful about how you craft the prompts being sent to the LLM.  The quality of the prompt directly correlates to the quality of the output. Consider using techniques like few-shot learning, chain-of-thought prompting, and prompt templates.
-   **Modular Tools:**
    - Design tools to be as reusable and composable as possible. This will allow you to create more complex agent workflows by combining simpler tools.
-   **Cost Tracking:**
    - When using paid LLM APIs, it's important to track the cost of each agent run. This can help you optimize your prompts and agent workflows to reduce costs.
-   **Monitoring:**
    - Monitor your agents in production to identify issues and performance bottlenecks.  Implement logging and alerting to catch errors early.

By following these best practices, you can develop robust, maintainable, and efficient AI agent applications with smolagents.