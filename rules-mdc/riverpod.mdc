---
description: Enforces Riverpod library best practices for Flutter applications.  This rule provides guidance on code organization, performance, testing, and common pitfalls when using Riverpod.
globs: *.dart
---
- **Core Principles:**
  - **Provider Initialization:**  Avoid initializing providers within widgets. Providers should initialize themselves, managing their own state and dependencies independently.  This ensures proper lifecycle management and prevents unexpected behavior during widget rebuilds.  Specifically, do not initialize providers inside `build()` methods or widget constructors.
  - **Provider Usage:** Primarily use `Provider` as the foundation for your state management.  Avoid legacy solutions like `ChangeNotifierProvider`, `StateNotifierProvider`, and `StateProvider` unless there are compelling reasons to use them, such as compatibility with existing code.  Prefer `StateProvider` or `NotifierProvider` (or `AsyncNotifierProvider`) for mutable state.
  - **Immutability:** Strive for immutability in your state management where possible.  Immutable data structures simplify debugging and prevent unexpected side effects. Use `freezed` or similar libraries to help enforce immutability.

- **Code Organization and Structure:**
  - **Directory Structure:**  Adopt a feature-based directory structure.  Group related files (providers, widgets, models, services) into dedicated directories representing specific features or modules. For example:

    
    lib/
    ├── features/
    │   ├── authentication/
    │   │   ├── data/
    │   │   │   ├── auth_repository.dart
    │   │   │   └── models/
    │   │   │       └── user.dart
    │   │   ├── presentation/
    │   │   │   ├── login_screen.dart
    │   │   │   ├── login_controller.dart
    │   │   │   └── providers.dart
    │   ├── home/
    │   │   ├── data/
    │   │   ├── presentation/
    │   │   └── providers.dart
    ├── core/
    │   ├── providers.dart  # Global providers
    │   └── services/
    │       └── api_service.dart
    └── main.dart
    
  - **File Naming Conventions:**  Use descriptive and consistent file names. For example:
    - `feature_name_screen.dart` (for widgets/screens)
    - `feature_name_service.dart` (for services)
    - `feature_name_repository.dart` (for repositories)
    - `feature_name_provider.dart` (for provider definitions)
  - **Module Organization:**  Break down your application into loosely coupled modules.  Each module should have a clear responsibility and a well-defined interface.  Use dependency injection to manage dependencies between modules, leveraging Riverpod's provider system.
  - **Component Architecture:** Design your UI with reusable components.  Extract common UI elements into separate widgets to promote code reuse and maintainability. Favor composition over inheritance.
  - **Code Splitting Strategies:**  Implement code splitting to reduce the initial load time of your application.  Use Flutter's lazy loading capabilities and consider splitting your application into smaller modules that can be loaded on demand.  Leverage `flutter_modular` or similar packages for advanced module management.

- **Common Patterns and Anti-patterns:**
  - **Design Patterns:**
    - **Repository Pattern:**  Abstract data access logic behind repositories.  Repositories handle data retrieval and persistence, isolating the rest of the application from the specifics of data sources (e.g., local storage, API).
    - **Service Pattern:**  Encapsulate business logic into services.  Services perform specific tasks or operations, such as authentication, data processing, or API integration.  Providers can depend on services to provide data or perform actions.
    - **Provider as State Holder:** Utilize Riverpod's providers to hold and manage application state.  Choose the appropriate provider type based on the state's characteristics (e.g., `StateProvider` for simple mutable state, `NotifierProvider` for complex state with side effects, `FutureProvider` or `StreamProvider` for asynchronous data).
  - **Recommended Approaches:**
    - **Asynchronous Data Handling:**  Use `FutureProvider` and `StreamProvider` to handle asynchronous data.  Display loading indicators while data is being fetched and handle errors gracefully.
    - **State Updates:** Use `state = newState` for `StateProvider` and `NotifierProvider`. Use `AsyncValue` to represent the different states of an asynchronous operation (loading, data, error).  Use `.when()` to handle different states in your UI.
    - **Side Effects:** Isolate side effects (e.g., API calls, database updates) in dedicated services or repositories.  Avoid performing side effects directly within widgets or providers.
  - **Anti-patterns and Code Smells:**
    - **Over-reliance on `ChangeNotifier`:** In most cases, `StateProvider` or `NotifierProvider` offer more straightforward and type-safe solutions than `ChangeNotifier`.
    - **Initializing Providers in Widgets:** This leads to provider recreation on every widget rebuild, impacting performance and potentially causing unexpected behavior.
    - **Directly Mutating State:**  Avoid directly mutating state managed by providers.  Instead, use the `state` setter for `StateProvider` or update the state within the `build` method of a `Notifier` / `AsyncNotifier`.
    - **Complex Logic in Widgets:**  Keep widgets focused on UI rendering. Move complex logic to services, repositories, or providers.
  - **State Management Best Practices:**
    - **Single Source of Truth:** Ensure that each piece of state has a single, authoritative source. Avoid duplicating state across multiple providers or widgets.
    - **Scoped Providers:** Use `providerScope` to scope providers to specific parts of the widget tree, avoiding unnecessary rebuilds. This also helps to manage the lifecycle of providers and prevent memory leaks.
  - **Error Handling Patterns:**
    - **Use `AsyncValue`'s `.when`:**  Handle different states of asynchronous operations, using `AsyncValue.when` to display loading indicators, data, and error messages.
    - **Global Error Handling:** Implement a global error handling mechanism to catch and log unexpected errors.  Consider using a `ProviderListener` to listen for errors and display appropriate error messages.

- **Performance Considerations:**
  - **Optimization Techniques:**
    - **`select`:** Use `.select` to only rebuild widgets when specific parts of a provider's state change.  This can significantly reduce unnecessary rebuilds.
    - **`Consumer` Widget:** Use `Consumer` widgets to rebuild only the parts of the widget tree that depend on a specific provider.
    - **`useProvider` Hook:**  Use the `useProvider` hook in functional components to access provider values efficiently.
  - **Memory Management:**
    - **Dispose Providers:** Ensure that providers are properly disposed of when they are no longer needed.  Use `ScopedProvider` to automatically dispose of providers when their scope is destroyed.
    - **Avoid Memory Leaks:**  Be mindful of potential memory leaks, especially when using streams or listeners.  Cancel subscriptions and dispose of resources when they are no longer needed.
  - **Rendering Optimization:**
    - **Minimize Widget Rebuilds:**  Use techniques like `const` constructors, `shouldRepaint`, and `useMemoized` to minimize unnecessary widget rebuilds.
    - **Optimize Image Loading:**  Use cached network images and optimize image sizes to improve rendering performance.
  - **Bundle Size Optimization:**
    - **Tree Shaking:**  Enable tree shaking to remove unused code from your application's bundle.  Use `flutter build apk --split-debug-info` for Android.
    - **Code Splitting:** Implement code splitting to reduce the initial load time of your application.
  - **Lazy Loading Strategies:** Load resources on demand to improve startup time and reduce memory consumption. Use `FutureProvider` and `StreamProvider` to load data lazily.

- **Security Best Practices:**
  - **Common Vulnerabilities:**
    - **Data Injection:**  Sanitize user input to prevent data injection attacks.
    - **Authentication Bypass:**  Implement robust authentication and authorization mechanisms to prevent unauthorized access.
    - **Insecure Data Storage:**  Protect sensitive data by encrypting it and storing it securely.
  - **Input Validation:**
    - **Validate User Input:**  Validate all user input on both the client-side and the server-side to prevent invalid data from being processed.
    - **Use Strong Data Types:**  Use strong data types to enforce data integrity and prevent type-related errors.
  - **Authentication and Authorization Patterns:**
    - **Secure Authentication:**  Use secure authentication protocols like OAuth 2.0 or JWT to authenticate users.
    - **Role-Based Authorization:**  Implement role-based authorization to control access to resources based on user roles.
  - **Data Protection Strategies:**
    - **Encryption:**  Encrypt sensitive data both in transit and at rest.
    - **Data Masking:**  Mask sensitive data in logs and error messages.
  - **Secure API Communication:**
    - **HTTPS:**  Use HTTPS to encrypt communication between the client and the server.
    - **API Keys:**  Protect API keys and store them securely.

- **Testing Approaches:**
  - **Unit Testing Strategies:**
    - **Test Provider Logic:**  Write unit tests to verify the logic within your providers.
    - **Mock Dependencies:**  Use mocking frameworks like `mockito` to isolate your providers from external dependencies.
  - **Integration Testing:**
    - **Test Provider Interactions:**  Write integration tests to verify how providers interact with each other and with other parts of the application.
    - **Test Data Flow:**  Test the data flow through the application to ensure that data is being processed correctly.
  - **End-to-End Testing:**
    - **Test User Flows:**  Write end-to-end tests to simulate user interactions and verify that the application behaves as expected.
    - **Automated UI Tests:**  Use tools like `flutter_driver` or `patrol` to automate UI tests.
  - **Test Organization:**
    - **Organize Tests by Feature:**  Organize your tests into directories that correspond to the features they test.
    - **Use Descriptive Test Names:**  Use descriptive test names that clearly explain what each test is verifying.
  - **Mocking and Stubbing:**
    - **Use Mocking Frameworks:** Use mocking frameworks like `mockito` to create mock objects for testing purposes.
    - **Create Test Doubles:** Create test doubles to replace real dependencies with simplified versions that are easier to test.

- **Common Pitfalls and Gotchas:**
  - **Frequent Mistakes:**
    - **Forgetting to dispose of providers:**  Leads to memory leaks.
    - **Incorrect provider scope:**  Leads to unexpected widget rebuilds.
    - **Not handling asynchronous errors:** Leads to unhandled exceptions.
  - **Edge Cases:**
    - **Provider lifecycle management:** Understand when providers are created and disposed of.
    - **Concurrency issues:**  Be aware of potential concurrency issues when working with asynchronous data.
  - **Version-Specific Issues:**
    - **Breaking changes:**  Be aware of breaking changes in new versions of Riverpod.
    - **Deprecated features:**  Avoid using deprecated features.
  - **Compatibility Concerns:**
    - **Flutter version compatibility:**  Ensure that your version of Riverpod is compatible with your version of Flutter.
    - **Package dependencies:** Resolve version conflicts between Riverpod and other packages.
  - **Debugging Strategies:**
    - **Use the Riverpod DevTools:**  Use the Riverpod DevTools to inspect provider state and dependencies.
    - **Logging:** Add logging statements to your providers to track their behavior.

- **Tooling and Environment:**
  - **Recommended Development Tools:**
    - **VS Code or IntelliJ IDEA:** Use a powerful IDE with Flutter and Dart support.
    - **Riverpod DevTools:**  Install the Riverpod DevTools extension for your IDE to inspect provider state and dependencies.
  - **Build Configuration:**
    - **Use `flutter build`:**  Use the `flutter build` command to build your application.
    - **Configure Build Flavors:**  Use build flavors to create different versions of your application for different environments (e.g., development, staging, production).
  - **Linting and Formatting:**
    - **Use `flutter analyze`:**  Use the `flutter analyze` command to analyze your code for potential problems.
    - **Use `dart format`:**  Use the `dart format` command to format your code according to the Dart style guide.
    - **Configure your linter**: Configure your linter to enforce Riverpod best practices.
  - **Deployment Best Practices:**
    - **Use CI/CD:** Use CI/CD pipelines to automate the build, test, and deployment process.
    - **Configure your environments**: Configure your different environments (development, staging, production) and their secrets properly. Using `flutter_dotenv` or similar.
  - **CI/CD Integration:**
    - **Integrate with GitHub Actions or GitLab CI:**  Integrate your CI/CD pipeline with GitHub Actions or GitLab CI.
    - **Automate Testing and Deployment:**  Automate the testing and deployment process to ensure that your application is always up-to-date.