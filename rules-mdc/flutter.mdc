---
description: Comprehensive guidelines and best practices for Flutter development, covering code organization, performance optimization, security, testing, and tooling.
globs: *.dart
---
- Adhere to the official Flutter style guide.  This forms the foundation for maintainable and consistent code.
- Utilize the latest stable version of Flutter, keeping up-to-date with new features and security patches. Review Flutter's breaking changes and migration guides during updates.

## 1. Code Organization and Structure

- **1.1 Directory Structure Best Practices:**
    - **`lib/` (Source Code):**
        -  Use a feature-based structure, grouping related components into modules.
        -  Example:
           
           lib/
           ├── auth/
           │   ├── models/
           │   ├── providers/
           │   ├── screens/
           │   ├── services/
           │   ├── widgets/
           │   └── auth.dart
           ├── home/
           │   ├── models/
           │   ├── screens/
           │   ├── widgets/
           │   └── home.dart
           ├── common/
           │   ├── models/
           │   ├── widgets/
           │   └── utils/
           ├── core/
           │   ├── services/
           │   ├── theme/
           │   └── utils/
           ├── main.dart
           └── app.dart
           
    - **`test/` (Tests):** Mirror the `lib/` structure for easy test discovery.
    - **`android/`, `ios/`, `web/`, `macos/`, `windows/`, `linux/` (Platform-Specific Code):**  Contain platform-specific native code. Limit direct modification unless necessary; utilize Flutter's platform channels.
    - **`assets/` (Assets):** Store images, fonts, and other static resources.  Organize subfolders by type (e.g., `images/`, `fonts/`, `data/`).  Use `pubspec.yaml` to declare assets.

- **1.2 File Naming Conventions:**
    -  Use `snake_case` for file names (e.g., `user_profile_screen.dart`).
    -  For classes within a file, the file name typically reflects the main class it contains.  Example: `user_profile_screen.dart` containing `UserProfileScreen`.
    -  Exceptions: Grouping multiple related enums, typedefs, or small helper functions into a single file is acceptable if it improves clarity.

- **1.3 Module Organization:**
    -  A module encapsulates a specific feature or functionality.  Modules should have well-defined interfaces and minimize dependencies on other modules.
    -  Implement a layered architecture within each module (e.g., UI, business logic, data access).
    -  Consider using packages for large, independent features to promote reusability across projects.

- **1.4 Component Architecture:**
    -  Favor a component-based architecture using Flutter widgets.  Break down complex UIs into smaller, reusable widgets.
    -  Separate presentation logic from business logic.
    -  Widgets should be pure functions of their input data (state).
    -  Follow the principles of Single Responsibility Principle (SRP) for widget design.

- **1.5 Code Splitting Strategies:**
    -  **Deferred Loading:** Load features on demand to reduce initial app size.
    -  **Route-Based Splitting:** Split code based on app routes.
    -  **Feature-Based Splitting:** Split code based on features.
    -  Use the `dart:ui` library's `loadFontFromList` or `ImageProvider.loadBuffer`  to load font or image resources dynamically.

## 2. Common Patterns and Anti-patterns

- **2.1 Design Patterns Specific to Flutter:**
    -  **BLoC (Business Logic Component):** Separates business logic from the UI, making the code more testable and maintainable.
    -  **Provider:** Simple dependency injection and state management solution.
    -  **Riverpod:** An improved version of Provider with compile-time safety.
    -  **GetX:** A microframework that provides state management, dependency injection, and route management.
    -  **MVVM (Model-View-ViewModel):** Another pattern for separating concerns.  Often used with reactive programming.
    -  **Redux/Flux:** For predictable state management, especially in complex applications.
    -  **InheritedWidget:** Implicit dependency injection for theming and configuration.

- **2.2 Recommended Approaches for Common Tasks:**
    -  **State Management:** Choose a state management solution that fits the complexity of the app.
    -  **Networking:** Use the `http` package or `dio` for making API requests.
    -  **Asynchronous Operations:** Use `async/await` for handling asynchronous operations.
    -  **Data Persistence:** Use `shared_preferences` for simple data storage or SQLite (using packages like `sqflite`) or NoSQL databases (using packages like `hive` or `isar`) for structured data.
    -  **Navigation:** Use `go_router` or `auto_route` for type-safe navigation.
    -  **Form Handling:** Use `Form` widget with `TextFormField` and validators.

- **2.3 Anti-patterns and Code Smells to Avoid:**
    -  **Massive Widgets:** Widgets with too much logic or UI code. Break them down into smaller, reusable components.
    -  **Logic in Widgets:** Avoid putting business logic directly inside widgets.
    -  **Deeply Nested Widgets:** Can lead to performance issues and difficult-to-read code.  Simplify the widget tree.
    -  **Unmanaged State:** Forgetting to dispose of resources like `StreamSubscription` or `AnimationController` leading to memory leaks.
    -  **Hardcoded Values:** Avoid hardcoding values like colors, sizes, and strings in the code. Use constants or theme data.
    -  **Ignoring Errors:** Not handling exceptions properly can lead to unexpected crashes.  Use `try-catch` blocks and logging.

- **2.4 State Management Best Practices:**
    -  Choose a state management solution that fits the complexity of the app.
    -  Keep the state as close to where it is needed as possible. Avoid global state for everything.
    -  Use immutable data structures to prevent unexpected state changes.
    -  Separate state from UI components to improve testability.
    -  Manage side effects properly.
    -  Consider reactive programming with streams for complex state transitions.

- **2.5 Error Handling Patterns:**
    -  Use `try-catch` blocks to handle exceptions.
    -  Implement custom error classes for specific error scenarios.
    -  Log errors to a file or remote service for debugging.
    -  Show user-friendly error messages.
    -  Use `ErrorWidget` to display custom error screens.
    -  Handle asynchronous errors using `Future.catchError` or `Stream.handleError`.

## 3. Performance Considerations

- **3.1 Optimization Techniques:**
    -  **Avoid Unnecessary Widget Rebuilds:** Use `const` constructors for immutable widgets, `shouldRebuild` method in `StatefulWidget`, and `ValueKey` for widgets that change position in a list.
    -  **Minimize `setState` Calls:** Use state management solutions to optimize state updates.
    -  **Use `ListView.builder` or `GridView.builder`:**  For large lists or grids, build widgets lazily.
    -  **Use `RepaintBoundary`:**  Isolate parts of the UI that don't need to be repainted often.
    -  **Use `Opacity` and `Clip` Sparingly:**  These operations can be expensive.
    -  **Use `Transform` carefully:** transforms can break batching and cause additional draw calls.

- **3.2 Memory Management:**
    -  Dispose of resources like `StreamSubscription`, `AnimationController`, and `TextEditingController` in the `dispose` method.
    -  Avoid creating unnecessary objects.
    -  Use the `dart:developer` package's memory profiling tools to identify memory leaks.
    -  Minimize the use of global variables and static fields.

- **3.3 Rendering Optimization:**
    -  Use the Flutter Performance Overlay to identify performance bottlenecks.
    -  Reduce the complexity of the widget tree.
    -  Optimize image loading and caching.
    -  Avoid using custom paint operations unless necessary.

- **3.4 Bundle Size Optimization:**
    -  Use `flutter build apk --split-per-abi` or `flutter build appbundle` to split the APK/AAB by ABI (Application Binary Interface).
    -  Remove unused assets and code.
    -  Compress images.
    -  Use code obfuscation and minification.
    -  Use deferred loading for infrequently used features.

- **3.5 Lazy Loading Strategies:**
    -  **Image Lazy Loading:** Load images only when they are visible on the screen.
    -  **Data Lazy Loading:** Load data in chunks as the user scrolls.
    -  Use the `VisibilityDetector` package to detect when a widget becomes visible.
    -  Use pagination or infinite scrolling for large datasets.

## 4. Security Best Practices

- **4.1 Common Vulnerabilities and How to Prevent Them:**
    -  **Data Injection:** Sanitize user input to prevent SQL injection, XSS, and other injection attacks.
    -  **Sensitive Data Storage:** Avoid storing sensitive data in plain text. Use encryption and secure storage mechanisms.
    -  **Insecure API Communication:** Use HTTPS for all API communication.
    -  **Code Tampering:** Use code obfuscation to make it harder to reverse engineer the app.
    -  **Man-in-the-Middle Attacks:** Implement certificate pinning to prevent MITM attacks.

- **4.2 Input Validation:**
    -  Validate all user input on both the client and server sides.
    -  Use regular expressions or custom validation logic to enforce data constraints.
    -  Encode data properly before displaying it in the UI.

- **4.3 Authentication and Authorization Patterns:**
    -  Use secure authentication protocols like OAuth 2.0 or OpenID Connect.
    -  Implement multi-factor authentication (MFA) for added security.
    -  Use role-based access control (RBAC) to restrict access to sensitive data and functionality.
    -  Store authentication tokens securely.

- **4.4 Data Protection Strategies:**
    -  Encrypt sensitive data at rest and in transit.
    -  Use secure storage mechanisms like the Keychain (iOS) or Keystore (Android).
    -  Follow the principle of least privilege when granting access to data.

- **4.5 Secure API Communication:**
    -  Use HTTPS for all API communication.
    -  Implement proper authentication and authorization.
    -  Validate API responses.
    -  Rate limit API requests to prevent abuse.

## 5. Testing Approaches

- **5.1 Unit Testing Strategies:**
    -  Test individual functions, classes, and widgets in isolation.
    -  Use mock objects to isolate the code under test from its dependencies.
    -  Write tests for all critical business logic.

- **5.2 Integration Testing:**
    -  Test the interaction between different parts of the app.
    -  Test the integration with external services like APIs and databases.

- **5.3 End-to-End Testing:**
    -  Test the entire app from start to finish.
    -  Simulate user interactions to ensure that the app works as expected.

- **5.4 Test Organization:**
    -  Create a `test/` directory that mirrors the `lib/` directory structure.
    -  Use descriptive test names.
    -  Keep tests small and focused.

- **5.5 Mocking and Stubbing:**
    -  Use mocking frameworks like `mockito` to create mock objects.
    -  Use stubbing to replace external dependencies with predefined values.
    -  Avoid over-mocking, as it can make tests less effective.

## 6. Common Pitfalls and Gotchas

- **6.1 Frequent Mistakes Developers Make:**
    -  Not disposing of resources.
    -  Ignoring errors.
    -  Hardcoding values.
    -  Using `setState` excessively.
    -  Creating massive widgets.
    -  Not validating user input.
    -  Over-complicating the state management.

- **6.2 Edge Cases to Be Aware Of:**
    -  Network connectivity issues.
    -  Device orientation changes.
    -  Background app state.
    -  Low memory conditions.
    -  Localization and internationalization.

- **6.3 Version-Specific Issues:**
    -  Be aware of breaking changes in new Flutter releases.
    -  Test the app on different Flutter versions to ensure compatibility.
    -  Use version constraints in `pubspec.yaml` to specify the required Flutter version.

- **6.4 Compatibility Concerns:**
    -  Test the app on different devices and operating systems.
    -  Consider accessibility for users with disabilities.
    -  Follow platform-specific guidelines for UI and functionality.

- **6.5 Debugging Strategies:**
    -  Use the Flutter DevTools for debugging and profiling.
    -  Use logging to track down errors.
    -  Use breakpoints to step through the code.
    -  Use the Flutter Inspector to inspect the widget tree.

## 7. Tooling and Environment

- **7.1 Recommended Development Tools:**
    -  Visual Studio Code or Android Studio.
    -  Flutter DevTools.
    -  Android Emulator or iOS Simulator.
    -  Git for version control.

- **7.2 Build Configuration:**
    -  Use `flutter build` to build the app for different platforms.
    -  Configure build settings in `pubspec.yaml`.
    -  Use different build configurations for development, staging, and production.

- **7.3 Linting and Formatting:**
    -  Use `flutter_lints` package for linting.
    -  Use `dart format` or Prettier for code formatting.
    -  Configure the IDE to automatically format the code on save.

- **7.4 Deployment Best Practices:**
    -  Follow the deployment guidelines for each platform.
    -  Use code signing to ensure the authenticity of the app.
    -  Use version control to manage releases.
    -  Monitor the app for crashes and errors after deployment.

- **7.5 CI/CD Integration:**
    -  Use a CI/CD tool like GitHub Actions, GitLab CI, or Jenkins to automate the build, test, and deployment process.
    -  Configure the CI/CD pipeline to run linting, formatting, and testing.
    -  Automate the release process to the app stores.

This document provides comprehensive guidelines and best practices for Flutter development.  Following these guidelines will help you write maintainable, performant, and secure Flutter apps.