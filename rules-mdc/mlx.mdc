---
description: This rule provides comprehensive best practices for the MLX library, covering code organization, performance, security, testing, and common pitfalls. It aims to promote consistent, efficient, and maintainable code when working with MLX on Apple platforms.
globs: *.swift
---
# MLX Library Best Practices

This document outlines the recommended best practices and coding standards for developing applications using the MLX library on Apple platforms, primarily using Swift. Following these guidelines will help ensure code quality, maintainability, performance, and security.

## 1. Code Organization and Structure

A well-structured project is crucial for scalability and maintainability. Here's how to organize your MLX projects effectively:

### 1.1. Directory Structure

Adopt a modular directory structure that separates concerns and promotes reusability. A typical project structure might look like this:


MyMLXProject/
├── Data/
│   ├── Datasets.swift         # Classes/structs for handling datasets
│   ├── Preprocessing.swift    # Data preprocessing utilities
├── Models/
│   ├── ModelDefinitions.swift # MLX Model Definitions (structs/classes)
│   ├── Layers.swift           # Custom layers for your models
│   ├── LossFunctions.swift    # Loss functions for training
├── Training/
│   ├── Trainer.swift          # Training loop and optimization logic
│   ├── Metrics.swift          # Evaluation metrics
├── Inference/
│   ├── InferenceEngine.swift  # Model inference and prediction code
├── Utilities/
│   ├── Helpers.swift          # Helper functions and extensions
│   ├── Configuration.swift   # Project configuration settings
├── UI/
│   ├── ViewControllers.swift  # UI-related code (if applicable)
│   ├── Views.swift           # Custom UI views
├── Tests/
│   ├── UnitTests.swift        # Unit tests for individual components
│   ├── IntegrationTests.swift # Integration tests for the application
├── MyMLXProject.xcodeproj   # Xcode project file
├── Podfile                   # CocoaPods dependencies (if using)
├── Package.swift             # Swift Package Manager manifest (if using)


### 1.2. File Naming Conventions

- Use descriptive and consistent names for your files.
- Follow the `PascalCase` convention for Swift files (e.g., `ModelDefinitions.swift`).
- Name files according to the primary type or functionality they contain.
- Use suffixes like `View`, `Controller`, `Model`, `Helper`, `Manager` to clearly indicate the role of the file.

### 1.3. Module Organization

- Organize your code into logical modules or Swift packages using Swift Package Manager (SPM).
- Each module should encapsulate a specific set of functionalities, such as data processing, model definitions, or training logic.
- Modules should have well-defined interfaces and minimize dependencies between them to promote reusability and testability.

### 1.4. Component Architecture

Consider using established architectural patterns like MVVM (Model-View-ViewModel) or VIPER (View-Interactor-Presenter-Entity-Router), especially for applications with a UI. However, even in non-UI projects:

- **Data Layer**: Responsible for data loading, preprocessing, and storage.  This could interface with Core Data, files, or network resources.
- **Model Layer**:  Holds the MLX model definitions, custom layers, and any associated logic.  Focus on making these types lightweight and easily serializable (if needed).
- **Service Layer**:  Manages model training, inference, and evaluation. This layer orchestrates the interaction between the data and model layers.
- **Utility Layer**: Contains helper functions, extensions, and common utilities used throughout the project.

### 1.5. Code Splitting Strategies

- Break down large files into smaller, more manageable units based on functionality or responsibility.
- Use extensions to logically group related methods within a class or struct (e.g., `// MARK: - Data Loading`).
- Employ protocols to define clear interfaces between components and enable loose coupling.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

- **Factory Pattern**: Use factory methods or classes to create instances of MLX models, layers, or optimizers, especially when dealing with complex configurations or dependencies.
- **Strategy Pattern**: Implement different optimization algorithms or loss functions as strategies that can be swapped in and out during training.
- **Observer Pattern**: Employ the observer pattern to notify components about training progress or model updates.

### 2.2. Recommended Approaches for Common Tasks

- **Data Loading**: Use `MLXData` or custom data loaders to efficiently load and preprocess your training data. Consider using memory mapping for large datasets.
- **Model Definition**: Define your models using `MLX` layers and sequential models. Leverage functional programming techniques for building complex model architectures.
- **Training Loop**: Implement a well-structured training loop that iterates over the data, performs forward and backward passes, updates model parameters, and evaluates performance.
- **Inference**: Use `MLXModel.predict` or custom inference engines to make predictions on new data.
- **Evaluation**:  Implement robust evaluation metrics and visualizations to monitor model performance and identify potential issues.

### 2.3. Anti-Patterns and Code Smells

- **Massive View Controllers**: Avoid placing all MLX-related logic directly within view controllers. Delegate tasks to service layers or dedicated MLX components.
- **Global State**: Minimize the use of global variables or singletons to manage MLX model instances or training data.  Instead, pass these objects as dependencies.
- **Hardcoded Values**: Avoid hardcoding model hyperparameters or data paths. Use configuration files or dependency injection to manage these values.
- **Ignoring Errors**:  Handle potential errors during data loading, model training, or inference gracefully, rather than ignoring or suppressing them.
- **Synchronous Operations on the Main Thread**: Offload time-consuming operations like model training or inference to background threads to prevent blocking the main thread and freezing the UI.

### 2.4. State Management

- For simple applications, pass state directly between components or use a simple state container object.
- For more complex applications, consider using state management frameworks like Combine or SwiftUI's `@StateObject` and `@EnvironmentObject`. Or a dedicated state management framework for ML models.
- Store model parameters and training progress in a dedicated state object and update it during the training loop.
- Make sure MLX-related state is properly handled and persisted across application lifecycle events (e.g., app termination, backgrounding).

### 2.5. Error Handling

- Use Swift's `Error` protocol and `do-catch` blocks to handle potential errors during data loading, model training, or inference.
- Create custom error types to represent specific MLX-related errors.
- Log error messages with sufficient context to aid debugging.
- Provide user-friendly error messages or feedback to the user when appropriate.

## 3. Performance Considerations

ML workloads are inherently performance-sensitive. Optimize your MLX code for speed and efficiency:

### 3.1. Optimization Techniques

- **Vectorization**: Leverage `MLX`'s vectorized operations to perform computations on entire arrays or matrices rather than iterating over individual elements.
- **Data Type Optimization**: Use the appropriate data types for your model parameters and training data (e.g., `Float16` or `Float32` instead of `Float64`).
- **Memory Optimization**: Minimize memory allocations and deallocations during the training loop. Re-use buffers and avoid unnecessary data copies.
- **Graph Compilation**:  If MLX supports it, compile your model graph to optimize its execution. This can significantly improve inference speed.
- **GPU Acceleration**: Ensure your MLX code is running on the GPU for faster computations (if available).  Verify GPU usage using system monitoring tools.

### 3.2. Memory Management

- Be mindful of memory usage, especially when dealing with large datasets or complex models.
- Use memory profiling tools to identify memory leaks or excessive memory allocations.
- Release unused memory explicitly when possible.
- Consider using techniques like data streaming or batch processing to reduce memory footprint.

### 3.3. Rendering Optimization (If Applicable)

-  MLX is focused on numerical computation, but if you are visualizing outputs, use techniques like Metal shaders to optimize rendering if you are drawing visualizations of ML data.

### 3.4. Bundle Size Optimization

- If using dependencies, consider the impact of external dependencies on the application bundle size.
- Use code stripping and dead code elimination to remove unused code and resources.
- Compress images, models, and other assets to reduce their size.

### 3.5. Lazy Loading

- Lazy load large models or datasets only when they are needed to reduce startup time and memory footprint.
- Use placeholder data or loading indicators while lazy-loading data in the background.

## 4. Security Best Practices

Even though MLX focuses on numerical computation, security is still important:

### 4.1. Common Vulnerabilities and Prevention

- **Model Poisoning**: Prevent malicious actors from injecting adversarial data into your training set to degrade model performance or inject biases.
  - **Mitigation**: Implement data validation and sanitization techniques. Monitor data sources and investigate anomalies.
- **Model Extraction**: Protect your trained models from being stolen or reverse-engineered by unauthorized parties.
  - **Mitigation**: Use model encryption or obfuscation techniques. Implement access controls and authentication mechanisms.
- **Adversarial Attacks**: Defend against adversarial attacks that can trick your models into making incorrect predictions.
  - **Mitigation**: Implement adversarial training techniques or input validation mechanisms.

### 4.2. Input Validation

- Validate all external inputs, including user data, API responses, and configuration files, to prevent malicious data from compromising your application or models.
- Sanitize input data to remove or escape potentially harmful characters or code.
- Use type checking and data validation libraries to ensure data conforms to expected formats and ranges.

### 4.3. Authentication and Authorization

- Implement robust authentication and authorization mechanisms to protect access to your models, training data, and API endpoints.
- Use secure password storage techniques like hashing and salting.
- Implement role-based access control (RBAC) to restrict access to sensitive resources based on user roles.

### 4.4. Data Protection

- Encrypt sensitive data at rest and in transit to prevent unauthorized access.
- Use secure storage options like the iOS Keychain to store sensitive information like API keys or passwords.
- Implement data masking or anonymization techniques to protect user privacy.

### 4.5. Secure API Communication

- Use HTTPS to encrypt communication between your application and external APIs or services.
- Implement certificate pinning to prevent man-in-the-middle attacks.
- Validate API responses to ensure data integrity and prevent data injection attacks.

## 5. Testing Approaches

Thorough testing is essential for ensuring the reliability and correctness of your MLX code:

### 5.1. Unit Testing

- Write unit tests to verify the functionality of individual components, such as model layers, loss functions, or data preprocessing utilities.
- Use mocking and stubbing techniques to isolate components and simulate dependencies.
- Test edge cases and boundary conditions to ensure robust behavior.

### 5.2. Integration Testing

- Write integration tests to verify the interaction between different components or modules.
- Test the data flow through the application pipeline, from data loading to model inference.
- Verify that the application integrates correctly with external APIs or services.

### 5.3. End-to-End Testing

- Write end-to-end tests to simulate user interactions and verify the overall application functionality.
- Test the user interface (if applicable) to ensure it behaves as expected.
- Verify that the application meets the specified performance and security requirements.

### 5.4. Test Organization

- Organize your tests into separate test suites or folders based on the component or functionality being tested.
- Use descriptive names for your test methods to clearly indicate their purpose.
- Follow a consistent naming convention for your test files (e.g., `MyComponentTests.swift`).

### 5.5. Mocking and Stubbing

- Use mocking frameworks to create mock objects that simulate the behavior of dependencies during unit testing.
- Use stubbing techniques to provide pre-defined responses or values for dependencies during testing.
- Avoid over-mocking or stubbing, as it can make tests brittle and less reliable.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

- **Incorrect Data Shapes**: Ensure that your input data and model parameters have the correct shapes and dimensions. Incorrect shapes can lead to errors or unexpected behavior.
- **Gradient Vanishing/Exploding**: Be aware of gradient vanishing or exploding problems during training. Use techniques like gradient clipping or batch normalization to mitigate these issues.
- **Overfitting**: Monitor your model's performance on a validation set to prevent overfitting. Use regularization techniques or dropout to reduce overfitting.
- **Not Freezing the graph before production**: After training and validation, ensure that MLX is not attempting to recalculate the graph each time inference is called.  Freeze the graph before deploying to production.

### 6.2. Edge Cases

- **Handling Missing Data**: Implement strategies for handling missing data in your datasets. Consider using imputation techniques or creating custom data loaders.
- **Dealing with Imbalanced Datasets**: Address imbalanced datasets by using techniques like oversampling, undersampling, or weighted loss functions.
- **Handling Out-of-Memory Errors**: Be prepared to handle out-of-memory errors when dealing with large models or datasets. Reduce batch sizes, use memory mapping, or distribute training across multiple GPUs.

### 6.3. Version-Specific Issues

- Be aware of potential compatibility issues between different versions of MLX and other dependencies.
- Consult the MLX documentation and release notes for information on known issues or breaking changes.
- Use dependency management tools like CocoaPods or SPM to manage dependencies and ensure compatibility.

### 6.4. Compatibility Concerns

- Consider compatibility between MLX and other technologies used in your project, such as Core ML or Metal.
- Use feature detection or conditional compilation to handle different platform versions or hardware capabilities.

### 6.5. Debugging Strategies

- Use Xcode's debugger to step through your code and inspect variables.
- Use logging statements to track the execution flow and data values.
- Use visualization tools to inspect model architectures, data distributions, and training progress.
- Leverage MLX's error messages and stack traces to identify and resolve issues.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

- **Xcode**: Use Xcode as your primary IDE for developing MLX applications on Apple platforms.
- **Metal Performance Shaders**: Use Metal Performance Shaders (MPS) to optimize performance on Apple GPUs.
- **Instruments**: Use Instruments to profile your application's performance and identify bottlenecks.
- **TensorBoard**: Use TensorBoard to visualize model architectures, training progress, and evaluation metrics.

### 7.2. Build Configuration

- Configure your Xcode project to optimize build settings for performance (e.g., enable optimization levels, disable debugging symbols).
- Use build configurations (e.g., Debug, Release) to manage different build settings for development and production environments.
- Use environment variables to configure application behavior based on the environment.

### 7.3. Linting and Formatting

- Use SwiftLint to enforce coding style guidelines and identify potential code quality issues.
- Use SwiftFormat to automatically format your code according to a consistent style.
- Configure Xcode to automatically run SwiftLint and SwiftFormat during the build process.

### 7.4. Deployment

- Prepare your MLX models and data for deployment. Convert models to efficient formats (e.g., Core ML) if necessary.
- Use code signing and provisioning profiles to ensure secure application distribution.
- Deploy your application to the App Store or distribute it using enterprise deployment mechanisms.

### 7.5. CI/CD Integration

- Integrate your MLX project with a continuous integration and continuous delivery (CI/CD) system like Jenkins, Travis CI, or GitHub Actions.
- Configure CI/CD pipelines to automatically build, test, and deploy your application on every code change.
- Use automated testing frameworks to verify the correctness of your MLX code and models.

By following these best practices, you can develop robust, efficient, and secure MLX applications that leverage the power of machine learning on Apple platforms.