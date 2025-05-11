---
description: This rule outlines best practices for Hardhat development, covering code organization, security, testing, and performance. It aims to provide a comprehensive guide for developers working with the Hardhat Ethereum development environment.
globs: *.js,*.ts,*.sol,*.json
---
# Hardhat Best Practices and Coding Standards

This document provides a comprehensive guide to best practices for developing smart contracts and decentralized applications (dApps) using Hardhat. It covers code organization, security, testing, performance, and other essential aspects of Hardhat development.

## 1. Code Organization and Structure

### 1.1. Directory Structure Best Practices


hardhat-project/
├── contracts/          # Solidity smart contracts
│   ├── MyContract.sol
│   └── ...
├── scripts/            # Deployment and interaction scripts
│   ├── deploy.js
│   ├── interact.js
│   └── ...
├── test/               # Unit and integration tests
│   ├── MyContract.test.js
│   └── ...
├── hardhat.config.js   # Hardhat configuration file
├── package.json        # Node.js package file
├── README.md           # Project documentation
└── .gitignore          # Git ignore file


-   **contracts/:**  Store all Solidity smart contracts in this directory.  Consider subdirectories for different modules or features.
-   **scripts/:**  Keep deployment, interaction, and other utility scripts in this directory.  Organize scripts by function or deployment environment.
-   **test/:**  Place all unit and integration tests in this directory.  Mirror the `contracts/` directory structure for test files.
-   **hardhat.config.js:**  Configure Hardhat settings in this file, including compiler versions, network configurations, and task definitions. Consider using TypeScript.
-   **package.json:**  Manage project dependencies and scripts using npm or yarn.
-   **README.md:**  Provide clear and concise documentation for your project.
-   **.gitignore:** Exclude unnecessary files from source control (e.g., build artifacts, node_modules).

### 1.2. File Naming Conventions

-   **Solidity Files:** Use PascalCase with the `.sol` extension for Solidity contract files (e.g., `MyContract.sol`).
-   **JavaScript/TypeScript Files:** Use camelCase with the `.js` or `.ts` extension for JavaScript/TypeScript files (e.g., `deploy.js`, `myContract.test.ts`).
-   **Test Files:**  Use the `.test.js` or `.test.ts` suffix to identify test files (e.g., `MyContract.test.js`).

### 1.3. Module Organization

-   **Separate Concerns:**  Divide your contracts into logical modules or components based on their functionality.
-   **Interfaces:**  Use interfaces to define the public API of your contracts.
-   **Libraries:**  Create reusable libraries for common functionalities to avoid code duplication.
-   **Abstract Contracts:**  Use abstract contracts to define common logic and state variables for related contracts.

### 1.4. Component Architecture

-   **Modular Design:** Design contracts as independent, reusable components.
-   **Composition:** Combine smaller components to create more complex systems.
-   **Minimal Dependencies:** Reduce dependencies between components to improve maintainability and testability.

### 1.5. Code Splitting Strategies

-   **Separate Logic and Data:**  Keep contract logic separate from data storage to improve readability and maintainability.
-   **External Libraries:**  Utilize external libraries (e.g., OpenZeppelin) for common functionalities to reduce code size and improve security.
-   **Proxy Patterns:**  Consider using proxy patterns to enable upgradeability and code splitting.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns Specific to Hardhat/Solidity

-   **Ownable:**  Control access to privileged functions using the Ownable pattern (from OpenZeppelin).
-   **Pausable:**  Implement a Pausable contract to temporarily halt contract functionality in case of emergencies.
-   **Pull over Push:**  Favor pull-based payment mechanisms over push-based mechanisms to mitigate reentrancy attacks.
-   **Proxy patterns (e.g., UUPS, Transparent Proxy):** For contract upgrades.

### 2.2. Recommended Approaches for Common Tasks

-   **Deployment:**  Use Hardhat's deployment scripts and network configurations to automate contract deployment.
-   **Testing:**  Employ Hardhat's testing framework and Chai matchers for comprehensive unit and integration testing.
-   **Interaction:**  Utilize Hardhat's console and Ethers.js for interacting with deployed contracts.
-   **Verification:**  Verify contract source code on Etherscan to enhance transparency and trust.

### 2.3. Anti-patterns and Code Smells to Avoid

-   **Unchecked Arithmetic:**  Use SafeMath or Solidity 0.8+ to prevent integer overflow/underflow.
-   **Reentrancy:**  Protect against reentrancy attacks by using the Checks-Effects-Interactions pattern or reentrancy guards (from OpenZeppelin).
-   **Denial of Service (DoS):**  Avoid patterns that can lead to DoS attacks, such as unbounded loops or expensive operations.
-   **Timestamp Dependence:**  Avoid relying on block timestamps for critical logic, as they can be manipulated by miners.
-   **Insufficient Gas Limits:** Ensure functions have sufficient gas limits, especially for complex operations.
-   **Hardcoding Addresses/Values:** Use configuration files or environment variables for addresses and other configurable values.

### 2.4. State Management Best Practices

-   **Minimize State Variables:**  Reduce the number of state variables to minimize storage costs and improve performance.
-   **Use Appropriate Data Types:** Choose the most efficient data types for state variables (e.g., `uint256` instead of `uint`).
-   **Immutability:** Use `immutable` variables when the value is assigned at construction and never changed after.
-   **Constants:** Use `constant` variables for gas optimization when the value is known at compile time and never changed after.
-   **Events:**  Emit events to track state changes and enable off-chain monitoring.

### 2.5. Error Handling Patterns

-   **Require Statements:**  Use `require` statements to validate inputs and enforce preconditions.
-   **Revert Statements:**  Use `revert` statements to handle exceptional conditions and return informative error messages.
-   **Custom Errors:** Use custom errors (Solidity 0.8.4+) for gas optimization and detailed error reporting.
-   **Try/Catch (Solidity 0.8.16+):** Implement Try/Catch blocks when calling external contracts.

## 3. Performance Considerations

### 3.1. Optimization Techniques

-   **Gas Optimization:**  Minimize gas consumption to reduce transaction costs.
-   **Data Packing:**  Pack multiple small variables into a single storage slot to reduce storage costs.
-   **Short Circuiting:**  Use short-circuiting evaluation in logical expressions to avoid unnecessary computations.
-   **Assembly Optimization:**  Use inline assembly for performance-critical sections of code.
-   **Caching:** Cache values that are frequently accessed.

### 3.2. Memory Management

-   **Minimize Memory Usage:**  Reduce memory usage to avoid out-of-gas errors.
-   **Storage vs. Memory:**  Use storage for persistent data and memory for temporary data.
-   **Delete Unused Variables:** Delete variables in memory after use to free up space.

### 3.3. Rendering Optimization (If applicable - for frontend DApps)

-   **Lazy Loading:**  Load components and data only when they are needed.
-   **Virtualization:**  Use virtualization techniques to render large lists efficiently.
-   **Memoization:**  Memoize computationally expensive functions to avoid redundant calculations.

### 3.4. Bundle Size Optimization (If applicable - for frontend DApps)

-   **Code Splitting:**  Split the application code into smaller bundles to improve initial load time.
-   **Tree Shaking:**  Remove unused code from the bundle using tree shaking.
-   **Minification:**  Minify JavaScript and CSS files to reduce bundle size.

### 3.5. Lazy Loading Strategies (If applicable - for frontend DApps)

-   **Dynamic Imports:**  Use dynamic imports to load modules on demand.
-   **Intersection Observer:**  Use the Intersection Observer API to load resources when they become visible.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities and How to Prevent Them

-   **Reentrancy:** Use Checks-Effects-Interactions pattern. Employ reentrancy guards (@openzeppelin/contracts/security/ReentrancyGuard.sol).
-   **Integer Overflow/Underflow:** Use SafeMath or Solidity 0.8+.
-   **Denial of Service (DoS):**  Limit gas costs, avoid unbounded loops, and implement rate limiting.
-   **Timestamp Dependence:**  Avoid relying on block timestamps for critical logic.
-   **Front Running:**  Design contracts to be resilient to front-running attacks.
-   **Signature Replay:** Prevent signature replay attacks by using nonces or other unique identifiers.
-   **Cross-Site Scripting (XSS):**  Sanitize user inputs in frontend DApps to prevent XSS attacks.
-   **SQL Injection:**  Use parameterized queries to prevent SQL injection attacks in backend systems.
-   **Improper Access Control:**  Enforce strict access control policies to protect sensitive data and functions.

### 4.2. Input Validation

-   **Sanitize Inputs:**  Sanitize and validate all user inputs to prevent malicious data from entering the system.
-   **Check Data Types:**  Verify that inputs are of the expected data types.
-   **Limit Input Length:**  Restrict the length of input strings to prevent buffer overflows.
-   **Regular Expressions:**  Use regular expressions to validate complex input patterns.

### 4.3. Authentication and Authorization Patterns

-   **Role-Based Access Control (RBAC):**  Implement RBAC to manage user permissions.
-   **Attribute-Based Access Control (ABAC):**  Use ABAC for more fine-grained access control.
-   **Multi-Factor Authentication (MFA):**  Implement MFA to enhance security.
-   **OAuth:**  Use OAuth for secure delegation of access to third-party applications.

### 4.4. Data Protection Strategies

-   **Encryption:**  Encrypt sensitive data at rest and in transit.
-   **Hashing:**  Use hashing to store passwords securely.
-   **Salting:**  Salt passwords to prevent rainbow table attacks.
-   **Key Management:**  Implement secure key management practices.

### 4.5. Secure API Communication (If applicable - for backend systems)

-   **HTTPS:**  Use HTTPS for all API communication.
-   **API Keys:**  Use API keys to authenticate API requests.
-   **Rate Limiting:**  Implement rate limiting to prevent abuse.
-   **Input Validation:** Validate and sanitize all API inputs.

## 5. Testing Approaches

### 5.1. Unit Testing Strategies

-   **Test Driven Development (TDD):**  Write tests before writing code.
-   **Black Box Testing:**  Test the functionality of the contract without knowledge of the internal implementation.
-   **White Box Testing:**  Test the internal implementation of the contract, including branches and loops.
-   **Coverage Analysis:**  Use coverage analysis tools to ensure that all code is covered by tests.

### 5.2. Integration Testing

-   **Test Contract Interactions:**  Test the interactions between multiple contracts.
-   **Test External Dependencies:**  Test the integration with external dependencies, such as oracles or other smart contracts.

### 5.3. End-to-End Testing (If applicable - for frontend DApps)

-   **Simulate User Flows:**  Simulate real-world user flows to test the entire application stack.
-   **Automated Testing:**  Use automated testing tools to run end-to-end tests regularly.

### 5.4. Test Organization

-   **Arrange-Act-Assert:**  Organize tests using the Arrange-Act-Assert pattern.
-   **Descriptive Test Names:**  Use descriptive names for tests to clearly indicate their purpose.
-   **Test Suites:**  Group related tests into test suites.

### 5.5. Mocking and Stubbing

-   **Mock External Dependencies:**  Use mocking to isolate contracts from external dependencies during testing.
-   **Stub Function Calls:**  Use stubbing to replace function calls with predefined values or behaviors.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes Developers Make

-   **Incorrectly handling gas costs**
-   **Failing to validate inputs**
-   **Ignoring security best practices**
-   **Not writing enough tests**
-   **Misunderstanding Ethereum's execution model**
-   **Not using the latest compiler version**

### 6.2. Edge Cases to Be Aware Of

-   **Integer overflow/underflow**
-   **Reentrancy attacks**
-   **Denial-of-service attacks**
-   **Front-running attacks**
-   **Signature replay attacks**

### 6.3. Version-Specific Issues

-   **Solidity compiler bugs:** Be aware of known bugs in the Solidity compiler and use appropriate workarounds.
-   **Hardhat version compatibility:** Ensure that Hardhat and its plugins are compatible with each other.

### 6.4. Compatibility Concerns

-   **EVM version compatibility:**  Consider the target EVM version when developing contracts.
-   **Web3.js/Ethers.js compatibility:** Ensure compatibility with the chosen JavaScript library for interacting with the blockchain.

### 6.5. Debugging Strategies

-   **Console.log:**  Use `console.log` statements to debug code.
-   **Hardhat console:** Use the Hardhat console for interactive debugging.
-   **Truffle debugger:**  Use the Truffle debugger for more advanced debugging.
-   **Remix IDE:** Use Remix IDE for online debugging.
-   **Solidity stack traces:**  Use Solidity stack traces to identify the source of errors.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

-   **Hardhat:**  Ethereum development environment.
-   **Visual Studio Code:**  Code editor with Solidity support.
-   **Remix IDE:**  Online Solidity IDE.
-   **Etherscan:**  Blockchain explorer.
-   **Ganache:**  Local Ethereum blockchain.
-   **OpenZeppelin:** Secure smart contract libraries.

### 7.2. Build Configuration

-   **Hardhat configuration file:**  Configure Hardhat settings in the `hardhat.config.js` file.
-   **Compiler version:**  Specify the Solidity compiler version in the Hardhat configuration file.
-   **Optimization settings:** Configure optimization settings in the Hardhat configuration file.

### 7.3. Linting and Formatting

-   **Solhint:**  Solidity linter.
-   **Prettier:**  Code formatter.
-   **ESLint:**  JavaScript/TypeScript linter.

### 7.4. Deployment Best Practices

-   **Automated deployments:**  Use Hardhat's deployment scripts to automate contract deployment.
-   **Network configurations:**  Configure network settings in the Hardhat configuration file.
-   **Gas price estimation:** Estimate gas prices before deploying contracts.
-   **Contract verification:**  Verify contract source code on Etherscan after deployment.

### 7.5. CI/CD Integration

-   **Continuous integration:**  Integrate Hardhat with CI/CD systems (e.g., GitHub Actions, Jenkins) to automate testing and deployment.