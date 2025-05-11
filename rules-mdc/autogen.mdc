---
description: Provides coding standards, best practices, and guidance specific to the AutoGen library, covering aspects from code structure to security and performance.
globs: *.py
---
- Best practice 1: Employ Modular Design. Utilize AutoGen's multi-agent framework to create modular agents with specific roles. This enhances code reusability and simplifies the orchestration of complex workflows. Example: Create separate agents for data fetching, code execution, and response formatting.
- Best practice 2: Define Customizable Workflows. Leverage AutoGen's capabilities to define customizable workflows that integrate LLMs, human inputs, and tools. This allows for flexible agent interactions and enhances the performance of LLM applications. Implement a workflow manager to define and control agent interactions.
- Best practice 3: Ensure Code Quality and Consistency. Implement automatic code formatting and linting features provided by AutoGen to ensure consistent coding styles and practices. This reduces human error in code generation and maintains high code quality. Integrate linters like `pylint` or `flake8` in your CI/CD pipeline.
- Best practice 4: Integrate Human Feedback. Incorporate a Human Proxy Agent to facilitate human feedback in the workflow, ensuring that the AI systems remain aligned with user expectations and can adapt based on real-world input. Use AutoGen's `HumanProxyAgent` to gather and incorporate human feedback in the loop.
- Best practice 5: Use UV for dependency management for faster and more reliable installations.
- Best practice 6: Always use the latest stable version of Python (currently 3.12) for better performance and security.
- Best practice 7: Prefer using classes over standalone functions to encapsulate state and behavior, particularly for agents and tools.
- Best practice 8: Separate configuration from code using environment variables and configuration files to allow easy modification without altering the code base

## Library Information:
- Name: autogen
- Tags: ai, ml, llm, python, agent-framework, multi-agent

## Comprehensive Guide to AutoGen Best Practices

### 1. Code Organization and Structure:
   - **Directory Structure Best Practices:**
     - `src/`: Contains the core AutoGen application code.
     - `src/agents/`: Holds individual agent implementations (e.g., `code_writer_agent.py`, `data_analyst_agent.py`).
     - `src/workflows/`: Defines workflows or agent interaction patterns (e.g., `code_generation_workflow.py`).
     - `src/tools/`: Contains custom tools and utilities used by agents (e.g., `web_scraper.py`, `api_client.py`).
     - `config/`: Stores configuration files (e.g., `agent_config.yaml`, `llm_config.json`).
     - `tests/`: Contains unit and integration tests.
     - `docs/`: Holds documentation.
     - `examples/`: Example scripts and notebooks showcasing AutoGen's capabilities.
   - **File Naming Conventions:**
     - Use descriptive names with snake_case (e.g., `code_executor_agent.py`, `data_processing_tool.py`).
     - Configuration files should use `.yaml` or `.json` extensions (e.g., `agent_config.yaml`).
     - Test files should be named `test_<module_name>.py` (e.g., `test_code_executor.py`).
   - **Module Organization Best Practices:**
     - Group related functionalities into modules (e.g., `agents`, `workflows`, `tools`).
     - Use `__init__.py` files to define packages and control imports.
     - Implement clear and concise module-level documentation.
   - **Component Architecture Recommendations:**
     - Separate agents, tools, and workflows into distinct components.
     - Use interfaces or abstract base classes to define contracts between components.
     - Implement a central registry or dependency injection mechanism to manage component dependencies.
   - **Code Splitting Strategies:**
     - Split large agent implementations into smaller, more manageable classes or functions.
     - Use lazy loading or dynamic imports to load components only when needed.
     - Implement a plugin system to allow external components to be added to the application.

### 2. Common Patterns and Anti-patterns:
   - **Design Patterns:**
     - **Factory Pattern:**  Create agents and tools dynamically based on configuration or user input.
     - **Strategy Pattern:** Implement different agent behaviors or tool execution strategies based on the current context.
     - **Observer Pattern:**  Allow agents to subscribe to events or notifications from other agents or tools.
   - **Recommended Approaches for Common Tasks:**
     - **Code Generation:** Use AutoGen's code generation capabilities to automate boilerplate code creation or data transformation tasks. Example: Automatically generate data validation functions based on schema definitions.
     - **Data Analysis:** Implement agents that can analyze data, generate visualizations, and extract insights. Example: Create an agent that analyzes website traffic data and identifies potential areas for improvement.
     - **Workflow Orchestration:** Define workflows that automate complex multi-step tasks involving multiple agents and tools. Example: Implement a workflow that automatically generates, tests, and deploys code changes.
   - **Anti-patterns and Code Smells:**
     - **God Agent:** Avoid creating a single agent that performs too many tasks. Decompose complex tasks into smaller, more manageable agents.
     - **Tight Coupling:**  Minimize dependencies between agents and tools. Use interfaces or abstract base classes to decouple components.
     - **Code Duplication:**  Extract common functionalities into reusable functions or classes.
   - **State Management Best Practices:**
     - Use a central state management system to store and manage the application's state.
     - Implement versioning or snapshotting to track state changes over time.
     - Use immutable data structures to prevent unintended state modifications.
   - **Error Handling Patterns:**
     - Implement robust error handling mechanisms to catch and handle exceptions.
     - Use logging to record errors and debug issues.
     - Implement retry logic to handle transient errors.

### 3. Performance Considerations:
   - **Optimization Techniques:**
     - **Caching:**  Cache frequently accessed data to reduce latency and improve throughput.
     - **Parallelization:**  Use asynchronous programming or multithreading to parallelize tasks and improve performance.
     - **Code Optimization:**  Profile and optimize code to reduce execution time.
   - **Memory Management:**
     - Use generators or iterators to process large datasets in chunks.
     - Release resources as soon as they are no longer needed.
     - Avoid creating unnecessary copies of data.
   - **Bundle Size Optimization (if applicable):** N/A for a Python Library, but if creating web apps with autogen, use tree-shaking to remove unused code and minimize bundle size.
   - **Lazy Loading:** Load modules and components only when needed to reduce startup time and memory usage.

### 4. Security Best Practices:
   - **Common Vulnerabilities:**
     - **Prompt Injection:**  Sanitize user inputs to prevent malicious code from being injected into agent prompts.
     - **Unauthorized Access:** Implement access controls to restrict access to sensitive data and functionalities.
     - **Data Leaks:** Protect sensitive data from being exposed in logs or error messages.
   - **Input Validation:**
     - Validate all user inputs to ensure that they conform to the expected format and range.
     - Use allow lists to restrict the types of inputs that are allowed.
     - Sanitize inputs to remove potentially harmful characters or code.
   - **Authentication and Authorization:**
     - Implement authentication mechanisms to verify the identity of users or agents.
     - Use authorization policies to control access to resources and functionalities.
     - Use role-based access control (RBAC) to simplify access management.
   - **Data Protection:**
     - Encrypt sensitive data at rest and in transit.
     - Use secure storage mechanisms to protect data from unauthorized access.
     - Implement data masking or anonymization techniques to protect sensitive data in logs or error messages.
   - **Secure API Communication:**
     - Use HTTPS to encrypt API communication.
     - Authenticate and authorize API requests.
     - Validate API responses to prevent data tampering.

### 5. Testing Approaches:
   - **Unit Testing:**
     - Write unit tests for individual agents, tools, and workflows.
     - Use mocking or stubbing to isolate components during testing.
     - Test different scenarios and edge cases.
   - **Integration Testing:**
     - Write integration tests to verify the interaction between different components.
     - Test the integration with external systems or APIs.
     - Use test data that is representative of real-world data.
   - **End-to-End Testing:**
     - Write end-to-end tests to verify the overall functionality of the application.
     - Simulate user interactions and verify that the application behaves as expected.
     - Use automated testing frameworks to automate the testing process.
   - **Test Organization:**
     - Organize tests into separate directories or files based on the component being tested.
     - Use descriptive names for test functions or classes.
     - Follow a consistent testing convention.
   - **Mocking and Stubbing:**
     - Use mocking to replace external dependencies with mock objects.
     - Use stubbing to replace complex functionalities with simple stubs.
     - Use mocking frameworks to simplify the mocking process.

### 6. Common Pitfalls and Gotchas:
   - **Frequent Mistakes:**
     - Not properly configuring LLM parameters (e.g., temperature, max tokens).
     - Overly complex prompts that confuse the LLM.
     - Ignoring error messages and logs.
     - Not validating user inputs.
   - **Edge Cases:**
     - Handling unexpected LLM responses.
     - Dealing with rate limits or API outages.
     - Handling large datasets or complex workflows.
   - **Version-Specific Issues:**
     - Be aware of breaking changes in AutoGen updates.
     - Check the release notes for known issues or workarounds.
   - **Compatibility Concerns:**
     - Ensure that AutoGen is compatible with other libraries or frameworks being used in the project.
     - Check for conflicts between different versions of AutoGen.
   - **Debugging Strategies:**
     - Use logging to track the execution flow and identify errors.
     - Use a debugger to step through the code and inspect variables.
     - Use print statements to output intermediate results.

### 7. Tooling and Environment:
   - **Recommended Development Tools:**
     - Python IDE (e.g., VS Code, PyCharm).
     - Version control system (e.g., Git).
     - Dependency management tool (e.g., pip, uv).
   - **Build Configuration:**
     - Use a build system to automate the build process.
     - Configure the build system to run linters and tests.
     - Use a virtual environment to isolate dependencies.
   - **Linting and Formatting:**
     - Use linters to enforce coding style guidelines.
     - Use formatters to automatically format code.
     - Integrate linters and formatters into the development workflow.
   - **Deployment:**
     - Use a deployment platform to deploy the application to a production environment.
     - Configure the deployment platform to automatically scale the application.
     - Use a monitoring system to track the application's performance.
   - **CI/CD Integration:**
     - Use a CI/CD pipeline to automate the build, test, and deployment process.
     - Integrate the CI/CD pipeline with the version control system.
     - Use automated testing to ensure code quality.