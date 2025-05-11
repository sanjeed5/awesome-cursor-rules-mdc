---
description: Enforces Jetpack Compose best practices for code organization, performance, and maintainability. This rule provides guidelines for writing efficient and idiomatic Compose code.
globs: *.kt
---
- **Code Organization and Structure**
  - **Directory Structure:**
    - Organize composables by feature or screen. Each feature should have its own directory.
    - Example:
      
      app/
        src/
          main/
            java/
              com/example/app/
                feature_a/
                  FeatureAScreen.kt
                  FeatureAViewModel.kt
                  components/
                    FeatureAButton.kt
                    FeatureATextField.kt
                feature_b/
                  ...
      
  - **File Naming Conventions:**
    - Use PascalCase for composable function names (e.g., `MyComposable`).
    - Use descriptive names that clearly indicate the composable's purpose.
    - Name files after the primary composable they contain (e.g., `MyComposable.kt`).
  - **Module Organization:**
    - Separate UI code (composables) from business logic (ViewModels, repositories).  Use modules to enforce this separation.
    - Consider using feature modules for large applications to improve build times and maintainability.
  - **Component Architecture:**
    - Design composables as small, reusable components.
    - Follow the single responsibility principle: each composable should have a single, well-defined purpose.
    - Use a hierarchical component structure to build complex UIs from smaller, simpler components.
  - **Code Splitting Strategies:**
    - Break down large composables into smaller, more manageable parts.
    - Use inline functions judiciously.  Overuse can hurt performance.
    - Consider using `derivedStateOf` to only recompose when necessary based on derived state.

- **Common Patterns and Anti-patterns**
  - **Design Patterns:**
    - **State Hoisting:** Move state as high as possible in the composable tree to maximize reusability and testability.  Pass state down as parameters.
    - **Unidirectional Data Flow:**  Data flows down the composable tree, and events flow up. This makes it easier to reason about the state of the UI.
    - **Composition Local:** Use sparingly to provide implicit dependencies to composables.  Avoid overusing as it can make dependencies less explicit.
    - **ViewModel:** Use a ViewModel to hold UI state and handle business logic. This separates the UI from the data layer and makes the UI easier to test.
  - **Recommended Approaches:**
    - Use `remember` to cache expensive calculations and avoid unnecessary recompositions.
    - Use `mutableStateOf` or `rememberSaveable` to store UI state.
    - Use `LaunchedEffect` or `rememberCoroutineScope` for side effects (e.g., network requests, database access).
    - Use `animate*AsState` functions for smooth animations.
  - **Anti-patterns and Code Smells:**
    - **Reading State from Too High a Scope:** Avoid reading state too high in the composition tree, as this can cause unnecessary recompositions.
    - **Mutable State in Composables without remember:** If you're not using remember, every recomposition will create a new state, leading to unexpected behavior.
    - **Long Composable Functions:** Break large composables into smaller, more manageable parts.
    - **Hardcoded Values:** Avoid hardcoding values directly in composables. Use resources or constants instead.
    - **Unnecessary Recomposition:**  Profile your code to identify and eliminate unnecessary recompositions.  Tools like the Layout Inspector can help.
  - **State Management Best Practices:**
    - Choose the right state management solution for your needs (e.g., `remember`, `mutableStateOf`, `StateFlow`, `LiveData`).
    - Keep state as immutable as possible.  Use `copy()` to create new state objects instead of modifying existing ones.
    - Use `derivedStateOf` to derive state from other state objects.
  - **Error Handling Patterns:**
    - Use `try-catch` blocks to handle exceptions in composables.
    - Display error messages to the user in a clear and informative way.
    - Use a central error handling mechanism to log errors and prevent crashes.  Consider using a `Snackbar` or a dedicated error display composable.

- **Performance Considerations**
  - **Optimization Techniques:**
    - **`remember`:** Cache expensive calculations and resources.
    - **`derivedStateOf`:** Only recompose when derived state changes.
    - **`SnapshotFlow`:** Collect data from mutable state without recomposing.
    - **`CompositionLocalProvider`:** Provides an alternative way to pass data if recomposition is a bottleneck.  Use with caution as it can make dependencies implicit.
    - **`Skippable Composable Functions`:** Compose Compiler performs the best optimization for functions that are skippable. To allow skipping, a composable must meet the following criteria:
      - All parameters passed to the composable are stable.
      - The composable's result is the same given the same parameters.
    - **Inline Functions (with care):** Useful for small functions but can increase bytecode size if overused.
  - **Memory Management:**
    - Avoid creating large objects in composables.
    - Release resources when they are no longer needed.
    - Use `WeakReference` to avoid memory leaks.
  - **Rendering Optimization:**
    - Use `Modifier.drawBehind` and `Modifier.drawWithContent` for custom drawing.
    - Avoid overdraw by using `Modifier.clip` and `Modifier.background`.
    - Use `Spacer` to control layout instead of adding padding to multiple elements.
  - **Bundle Size Optimization:**
    - Use R8 to shrink and obfuscate your code.
    - Remove unused resources.
    - Use dynamic feature modules to deliver features on demand.
  - **Lazy Loading Strategies:**
    - Use `LazyColumn` and `LazyRow` to display large lists of data.
    - Use `rememberLazyListState` to persist the scroll position of lazy lists.

- **Security Best Practices**
  - **Common Vulnerabilities:**
    - **Input Validation:**  Sanitize user input to prevent injection attacks.
    - **Data Protection:**  Encrypt sensitive data at rest and in transit.
    - **Secure API Communication:**  Use HTTPS to encrypt communication between the app and the server.
  - **Input Validation:**
    - Validate user input to prevent injection attacks and other security vulnerabilities.
    - Use regular expressions or other validation techniques to ensure that input is in the expected format.
  - **Authentication and Authorization Patterns:**
    - Use a secure authentication and authorization mechanism to protect sensitive data and functionality.
    - Use OAuth 2.0 or OpenID Connect for authentication.
    - Use role-based access control (RBAC) for authorization.
  - **Data Protection Strategies:**
    - Encrypt sensitive data at rest and in transit.
    - Use a strong encryption algorithm.
    - Store encryption keys securely.
  - **Secure API Communication:**
    - Use HTTPS to encrypt communication between the app and the server.
    - Use certificate pinning to prevent man-in-the-middle attacks.

- **Testing Approaches**
  - **Unit Testing Strategies:**
    - Test composables in isolation using `ComposeTestRule`.
    - Verify that composables render correctly and handle user input as expected.
    - Use `composeTestRule.setContent { ... }` to set the content of the test.
    - Use `composeTestRule.onNodeWithText("...").performClick()` to simulate user interactions.
  - **Integration Testing:**
    - Test the interaction between different composables and ViewModels.
    - Use `Hilt` or `Koin` to inject dependencies into tests.
  - **End-to-end Testing:**
    - Test the entire application flow from the user's perspective.
    - Use `UIAutomator` or `Espresso` for end-to-end testing.
  - **Test Organization:**
    - Organize tests by feature or screen.
    - Create a separate test module for each feature module.
  - **Mocking and Stubbing:**
    - Use `Mockito` or `Mockk` to mock dependencies in unit tests.
    - Use `Fake` implementations for integration tests.

- **Common Pitfalls and Gotchas**
  - **Frequent Mistakes:**
    - Forgetting to use `remember` for state.
    - Reading state from too high a scope.
    - Not handling errors properly.
    - Not optimizing performance.
  - **Edge Cases:**
    - Handling different screen sizes and orientations.
    - Handling different locales and languages.
    - Handling different accessibility requirements.
  - **Version-Specific Issues:**
    - Be aware of breaking changes in new versions of Jetpack Compose.
    - Use the latest version of Jetpack Compose to take advantage of new features and bug fixes.
  - **Compatibility Concerns:**
    - Ensure that your app is compatible with different Android versions.
    - Use Jetpack Compose libraries that are compatible with your target Android version.
  - **Debugging Strategies:**
    - Use the Android Studio debugger to step through your code and inspect variables.
    - Use the Layout Inspector to inspect the composable tree and identify performance bottlenecks.
    - Use the Profiler to measure the performance of your app.

- **Tooling and Environment**
  - **Recommended Development Tools:**
    - Android Studio
    - Kotlin Compiler
    - Jetpack Compose Libraries
  - **Build Configuration:**
    - Use Gradle to manage dependencies and build your app.
    - Configure the Kotlin compiler to use the latest version of Jetpack Compose.
  - **Linting and Formatting:**
    - Use `ktlint` or `detekt` to enforce code style guidelines.
    - Use `Prettier` to format your code automatically.
  - **Deployment Best Practices:**
    - Use the Google Play Store to distribute your app.
    - Use a staged rollout to gradually release your app to users.
  - **CI/CD Integration:**
    - Use a CI/CD system to automate the build, test, and deployment process.
    - Use `GitHub Actions` or `CircleCI` for CI/CD.