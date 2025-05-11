---
description: This rule provides comprehensive guidelines for using the Zod library effectively, covering code organization, performance, security, and testing to ensure robust and maintainable type validation.
globs: *.ts?(x)
---
- **Organize Zod schemas logically for readability and maintainability.** Group related schemas together and structure your Zod code for improved clarity, especially in larger projects.
- **Compose and reuse related schemas to avoid repetition.** Use Zod's composition features (e.g., `z.intersection`, `z.union`, `z.extend`) to create reusable schema components and reduce redundancy.
- **Implement schema versioning for better management.** As your application evolves, version your schemas to handle data migrations and maintain compatibility with older data formats.
- **Use Zod for type-safe data validation and transformation.** Leverage Zod's capabilities for both validating and transforming data to ensure data integrity throughout your application.

### 1. Code Organization and Structure:

- **Directory Structure Best Practices:**
    - Consider grouping Zod schemas within dedicated directories (e.g., `schemas/`, `models/schemas/`).
    - Organize schemas by domain, feature, or data model.
    - Use subdirectories for complex schemas or schema families.
- **File Naming Conventions:**
    - Name schema files descriptively (e.g., `user.schema.ts`, `product.schema.ts`).
    - Use a consistent naming pattern throughout the project.
    - Consider including the data model name or the schema's purpose in the filename.
- **Module Organization:**
    - Export schemas as named exports from each module.
    - Create index files (e.g., `index.ts`) to re-export schemas from subdirectories for easier access.
    - Use clear and concise module names.
- **Component Architecture:**
    - If you are building React components, consider creating a `components/schemas/` directory where you house your schema related to specific components.
    - Use Zod to validate the props passed to React components using `z.infer` and `z.ZodType<Props>`
    - You can create a custom hook that handles validation with Zod and stores the parsed result in React state.
- **Code Splitting Strategies:**
    - For large schema definitions, split them into smaller, more manageable files.
    - Use Zod's composition features to combine these smaller schemas into larger, more complex schemas as needed.

### 2. Common Patterns and Anti-patterns:

- **Design Patterns Specific to Zod:**
    - **Schema Composition:** Use `z.intersection`, `z.union`, `z.extend`, and `z.optional` to combine and modify existing schemas.
    - **Schema Transformation:** Use `.transform` to modify data during validation.
    - **Custom Validation:** Use `.refine` and `.superRefine` for custom validation logic.
    - **Default Values:** Use `.default` to assign default values to schema properties.
- **Recommended Approaches for Common Tasks:**
    - **Form Validation:** Use Zod to validate form input data and display errors.
    - **API Request Validation:** Use Zod to validate incoming API request bodies.
    - **Data Serialization/Deserialization:** Use Zod to validate and transform data when serializing or deserializing.
- **Anti-patterns and Code Smells to Avoid:**
    - **Overly Complex Schemas:** Avoid creating schemas that are too complex or difficult to understand. Break them down into smaller, more manageable schemas.
    - **Ignoring Validation Errors:** Always handle validation errors and provide informative feedback to the user.
    - **Duplicated Schema Definitions:** Avoid duplicating schema definitions. Use schema composition to reuse existing schemas.
- **State Management Best Practices:**
    - When using Zod with state management libraries (e.g., Redux, Zustand), validate the state data using Zod schemas.
    - Use Zod to validate state updates before applying them to the state.
- **Error Handling Patterns:**
    - Use Zod's `.safeParse` method to handle validation errors gracefully.
    - Provide informative error messages to the user.
    - Log validation errors for debugging purposes.

### 3. Performance Considerations:

- **Optimization Techniques:**
    - **Schema Caching:** Cache frequently used schemas to avoid re-parsing them.
    - **Pre-compilation:** If possible, pre-compile schemas during build time to improve performance.
    - **Minimize Schema Complexity:** Keep schemas as simple as possible to reduce validation overhead.
- **Memory Management:**
    - Be mindful of the memory usage of large schemas, especially when dealing with large datasets.
    - Release unused schemas when they are no longer needed.
- **Bundle Size Optimization:**
    - Remove unused schemas and code from the bundle.
    - Use tree shaking to eliminate dead code.
- **Lazy Loading Strategies:**
    - Lazily load schemas that are not immediately needed.
    - Use code splitting to load schemas on demand.

### 4. Security Best Practices:

- **Common Vulnerabilities and How to Prevent Them:**
    - **Injection Attacks:** Prevent injection attacks by validating and sanitizing user input data.
    - **Cross-Site Scripting (XSS):** Prevent XSS attacks by encoding user input data before displaying it in the UI.
    - **Denial of Service (DoS):** Prevent DoS attacks by limiting the size and complexity of input data.
- **Input Validation:**
    - Validate all user input data using Zod schemas.
    - Enforce strict validation rules to prevent invalid or malicious data from entering the system.
- **Authentication and Authorization Patterns:**
    - Use Zod to validate user credentials during authentication.
    - Use Zod to validate authorization tokens and permissions.
- **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use secure storage mechanisms to protect data from unauthorized access.
- **Secure API Communication:**
    - Use HTTPS to encrypt API communication.
    - Validate API request and response data using Zod schemas.

### 5. Testing Approaches:

- **Unit Testing Strategies:**
    - Write unit tests for individual schemas to ensure they validate data correctly.
    - Test different input scenarios, including valid and invalid data.
    - Use mocking and stubbing to isolate schemas from external dependencies.
- **Integration Testing:**
    - Write integration tests to ensure that schemas work correctly with other parts of the application.
    - Test the interaction between schemas and data sources (e.g., databases, APIs).
- **End-to-End Testing:**
    - Write end-to-end tests to ensure that the entire application works correctly with Zod schemas.
    - Test the user interface and the data flow through the application.
- **Test Organization:**
    - Organize tests into separate directories based on the type of test (e.g., unit, integration, end-to-end).
    - Use descriptive test names to indicate the purpose of each test.
- **Mocking and Stubbing:**
    - Use mocking and stubbing to isolate schemas from external dependencies during testing.
    - Mock data sources and APIs to control the test environment.

### 6. Common Pitfalls and Gotchas:

- **Frequent Mistakes Developers Make:**
    - **Incorrect Schema Definitions:** Ensure that schema definitions accurately reflect the expected data format.
    - **Ignoring Validation Errors:** Always handle validation errors and provide informative feedback to the user.
    - **Overly Complex Schemas:** Avoid creating schemas that are too complex or difficult to understand.
- **Edge Cases to Be Aware Of:**
    - **Null and Undefined Values:** Handle null and undefined values correctly in schemas.
    - **Empty Strings and Arrays:** Handle empty strings and arrays appropriately.
    - **Date and Time Formats:** Use consistent date and time formats throughout the application.
- **Version-Specific Issues:**
    - Be aware of any version-specific issues or breaking changes in Zod.
    - Refer to the Zod documentation and release notes for information on compatibility and migration.
- **Compatibility Concerns:**
    - Ensure that Zod schemas are compatible with the data formats used by other systems or libraries.
    - Consider using a data serialization format (e.g., JSON, YAML) that is widely supported.
- **Debugging Strategies:**
    - Use Zod's `.safeParse` method to log validation errors for debugging purposes.
    - Use a debugger to step through the validation process and identify issues.

### 7. Tooling and Environment:

- **Recommended Development Tools:**
    - **TypeScript:** Use TypeScript to provide static typing for Zod schemas and data.
    - **VS Code:** Use VS Code with the Zod extension for improved code completion and validation.
- **Build Configuration:**
    - Configure the build process to transpile TypeScript code and bundle JavaScript code.
    - Use a module bundler (e.g., Webpack, Parcel, Rollup) to optimize the bundle size.
- **Linting and Formatting:**
    - Use a linter (e.g., ESLint) to enforce code style and best practices.
    - Use a code formatter (e.g., Prettier) to automatically format code.
- **Deployment Best Practices:**
    - Deploy the application to a production environment with appropriate security measures in place.
    - Monitor the application for errors and performance issues.
- **CI/CD Integration:**
    - Integrate Zod schemas into the CI/CD pipeline to automate validation and testing.
    - Run unit tests, integration tests, and end-to-end tests as part of the CI/CD process.