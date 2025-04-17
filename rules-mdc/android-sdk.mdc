---
description: This rule file provides comprehensive best practices for Android SDK development, encompassing code organization, performance, security, testing, and common pitfalls. It aims to guide developers in creating robust, maintainable, and efficient Android applications and libraries.
globs: *.java,*.kt,*.xml,*.gradle,*.properties
---
- MUST be under source control management (SCM). RECOMMENDED source control management is git
- If you intend your source repo to be public ensure you have received the necessary management and legal approval
- MUST use a .gitignore configuration tailored for Android projects (e.g., ignoring `local.properties`, `build/`, `.gradle/`)
- Source code repositories SHOULD follow the naming convention `[platform]-[library name]`, e.g., `android-analytics`, `java-datastore`
- MUST NOT restrict read access to source code repositories unnecessarily
- RECOMMEND to not have source file header comment blocks as these easily get out of date

## Code Organization and Structure

- SHOULD follow the standard project layout for Android projects. Key directories:
    - `src/main/java`: Java/Kotlin source code
    - `src/main/res`: Resources (layouts, drawables, strings, etc.)
    - `src/test/java`: Unit tests
    - `src/androidTest/java`: Instrumentation tests
- MUST prefix all exported resources (layout files, XML files, drawable IDs, color IDs, dimension IDs, string IDs, etc.) with a unique prefix for the library using `resourcePrefix` in the Gradle build file.  Example: `sug` for a Suggestion Module. This avoids naming conflicts when the library is used in other projects.
- Use package names that are descriptive and follow the reverse domain name convention (e.g., `com.example.myapp.feature`).
- Organize code into logical modules or features. Each module should have a clear responsibility and be loosely coupled with other modules.
- Use Kotlin for new projects. If working with Java, target Java 8 or higher to utilize modern language features.
- Consider using a multi-module project structure for larger projects to improve build times and code organization. Modules can represent different features, libraries, or build variants.
- Employ a layered architecture (e.g., presentation, domain, data) to separate concerns and improve testability.

### Directory Structure Best Practices:


root/
├── app/
│   ├── build.gradle
│   └── src/
│       ├── androidTest/
│       │   └── java/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── java/
│       │   │   └── com/example/app/
│       │   │       ├── MainActivity.kt
│       │   │       ├── model/
│       │   │       ├── view/
│       │   │       └── viewmodel/
│       │   └── res/
│       │       ├── layout/
│       │       ├── drawable/
│       │       ├── values/
│       │       └── mipmap-xxhdpi/
│       └── test/
│           └── java/
├── library_module/
│   ├── build.gradle
│   └── src/
│       ├── main/
│       │   ├── java/
│       │   └── res/
│       └── test/
├── build.gradle
├── settings.gradle


### File Naming Conventions:

- **Kotlin/Java:** `PascalCase` for class names (e.g., `MainActivity`), `camelCase` for variables and method names (e.g., `userName`, `calculateTotal`).
- **Layout XML:** `snake_case` (e.g., `activity_main.xml`). Use descriptive names reflecting the layout's purpose.
- **Drawable XML:** `snake_case` (e.g., `ic_launcher.xml`). Prefix icons with `ic_`, backgrounds with `bg_`, and other drawables accordingly.
- **Values XML:** `snake_case` (e.g., `strings.xml`, `colors.xml`, `dimens.xml`).
- **Gradle:** `build.gradle` (module-level), `settings.gradle` (project-level).

### Component Architecture Recommendations:

- MVVM (Model-View-ViewModel) is the recommended architecture pattern. Use Android Architecture Components (LiveData, ViewModel, Room) to implement it effectively.
- For complex UI logic, consider using a ViewBindingDelegate to simplify view access and reduce boilerplate.
- Use dependency injection (Hilt or Dagger) to manage dependencies and improve testability.

## Common Patterns and Anti-patterns

- Use `LiveData` and `Flow` for reactive data streams.
- Utilize coroutines for asynchronous operations to simplify threading and avoid callback hell.
- Avoid performing long-running operations on the main thread (UI thread). Use `Dispatchers.IO` or `Dispatchers.Default` with coroutines or `AsyncTask` for background tasks.
- Don't use `AsyncTask` if possible, prefer coroutines.
- Prefer `ConstraintLayout` over other layout types for complex UIs as it is more flexible and performant.
- Use `RecyclerView` for displaying large lists of data efficiently. Implement view holder pattern correctly for optimal performance.
- Apply data binding to reduce boilerplate code in UI updates.
- Use sealed classes for representing state to ensure exhaustive handling of all possible states.
- Implement proper error handling with try-catch blocks and report errors gracefully to the user.
- Log errors and exceptions using a logging framework (Timber) for debugging and analysis.

### Anti-patterns and Code Smells to Avoid:

- **God Classes:** Classes that are too large and have too many responsibilities. Break them down into smaller, more focused classes.
- **Spaghetti Code:** Code that is difficult to read and understand due to complex control flow. Refactor it to improve readability and maintainability.
- **Copy-Paste Programming:** Duplicating code across multiple locations. Extract the common code into reusable functions or classes.
- **Hardcoding Values:** Avoid hardcoding values directly in the code. Use resources (strings, colors, dimensions) for better maintainability and localization.
- **Leaking Context:** Holding onto `Activity` or `Context` instances for longer than necessary, leading to memory leaks.

### State Management Best Practices:

- Use `ViewModel` to store UI-related data that survives configuration changes.
- Utilize `SavedStateHandle` in `ViewModel` to persist data across process death.
- Consider using a state management library (e.g., Redux, MVI) for complex applications with complex state.
- Implement appropriate caching strategies to reduce network requests and improve performance.

### Error Handling Patterns:

- Use try-catch blocks to handle potential exceptions and prevent app crashes.
- Report errors gracefully to the user with informative error messages.
- Log errors and exceptions using a logging framework (Timber) for debugging and analysis.
- Implement retry mechanisms for transient errors (e.g., network connection issues).

## Performance Considerations

- Optimize image loading using libraries like Glide or Coil. Resize images to the required dimensions before loading them into `ImageViews`.
- Use efficient data structures (e.g., `SparseArray` instead of `HashMap` for integer keys).
- Avoid creating unnecessary objects, especially in loops or frequently called methods.
- Use `StringBuilder` instead of string concatenation for building strings efficiently.
- Minimize network requests and optimize data transfer using compression and pagination.
- Analyze app performance using profiling tools (Android Profiler) to identify bottlenecks.
- Utilize the LeakCanary library to detect memory leaks during development.
- Use ProGuard or R8 to shrink, obfuscate, and optimize the code for release builds.

### Memory Management Considerations:

- Avoid creating memory leaks by properly managing object lifecycles and releasing resources when they are no longer needed.
- Use `WeakReference` to hold references to objects that may be garbage collected to prevent memory leaks.
- Monitor memory usage using the Android Profiler and identify potential memory leaks.

### Rendering Optimization:

- Minimize overdraw by reducing the number of layers that are drawn on top of each other.
- Use hardware acceleration to improve rendering performance.
- Optimize custom views by implementing `onDraw()` method efficiently and avoiding unnecessary allocations.

### Bundle Size Optimization:

- Remove unused resources using `shrinkResources` in the Gradle build file.
- Use vector drawables instead of raster images for icons and other simple graphics.
- Enable code shrinking and obfuscation using ProGuard or R8.
- Compress images using tools like TinyPNG or ImageOptim.
- Use app bundles to deliver optimized APKs for different device configurations.
- Utilize dynamic feature modules to defer the download of non-essential features.

### Lazy Loading Strategies:

- Implement lazy loading for images and other resources that are not immediately visible on the screen.
- Use pagination or infinite scrolling to load data in chunks as the user scrolls down the list.

## Security Best Practices

- MUST use Kotlin or Java. Kotlin is RECOMMENDED
- SHOULD use static code analysis, such as detekt or SwiftLint to minimise errors and minimise debates and formatting fix-up commits in PRs. It is RECOMMENDED to integrate these tools with the CI workflow. SHOULD use quality tools available in common config: Android and iOS
- Android: MUST use the Android Support Annotations for public APIs. Non-Android Java code MUST use the Jetbrains Annotations for public APIs
- Secure API communication and proper permission handling are vital to protect user data.
- MUST use a `consumerProguardFiles` file if obfuscation breaks the libraries function.
- SHOULD NOT have unnecessary exported dependencies, in particular SHOULD NOT leak networking library to consumers. SHOULD NOT leak serialization library to consumers.  An exception to this rule are optional adapters for popular libraries.

### Common Vulnerabilities and Prevention:

- **SQL Injection:** Avoid using raw SQL queries. Use parameterized queries or an ORM like Room to prevent SQL injection attacks.
- **Cross-Site Scripting (XSS):** Sanitize user input to prevent XSS attacks.
- **Man-in-the-Middle (MITM) Attacks:** Use HTTPS for all network communication to encrypt data in transit and prevent MITM attacks.
- **Insecure Data Storage:** Encrypt sensitive data before storing it on the device.
- **Improper Platform Usage:** Ensure that permissions are properly requested and handled. Review and minimize the permissions your app requests.

### Input Validation:

- Validate all user input on both the client and server sides to prevent malicious data from being processed.
- Use regular expressions or validation libraries to enforce input constraints.

### Authentication and Authorization:

- Use secure authentication protocols (e.g., OAuth 2.0, OpenID Connect) for user authentication.
- Implement proper authorization mechanisms to control access to resources based on user roles.
- Store user credentials securely using hashing and salting.

### Data Protection:

- Encrypt sensitive data at rest using AES or other strong encryption algorithms.
- Securely transmit data over the network using HTTPS.
- Implement data masking or anonymization techniques to protect user privacy.

### Secure API Communication:

- Use HTTPS for all API communication.
- Verify the server's SSL certificate to prevent MITM attacks.
- Implement API rate limiting to prevent denial-of-service attacks.

## Testing Approaches

- All library (sub-)projects MUST generate coverage test reports (e.g. jacoco, Xcode/slather).
- Unit test coverage MUST be at least 80% and SHOULD be above 85%
- Unit tests MUST NOT rely on reflection to expose otherwise private/internal classes or members.
- Tests SHOULD use JUnit, and MAY use helper libraries such as Mockito or Robolectric

### Unit Testing Strategies:

- Write unit tests for all critical components, including `ViewModels`, repositories, and use cases.
- Use mocking frameworks (Mockito) to isolate units under test and mock dependencies.
- Follow the AAA (Arrange, Act, Assert) pattern for writing clear and concise unit tests.

### Integration Testing Approaches:

- Write integration tests to verify the interaction between different components, such as `ViewModels` and repositories.
- Use dependency injection to provide mock dependencies for integration tests.

### End-to-End Testing:

- Use UI testing frameworks (Espresso, UI Automator) to write end-to-end tests that simulate user interactions.

### Test Organization:

- Organize tests into separate directories based on the type of tests (unit, integration, UI).
- Use descriptive names for test classes and methods to clearly indicate what is being tested.

### Mocking and Stubbing:

- Use mocking frameworks (Mockito) to create mock objects for dependencies.
- Use stubbing to define the behavior of mock objects for specific test cases.

## Common Pitfalls and Gotchas

- Frequent mistakes developers make when using android-sdk
- Edge cases to be aware of when using android-sdk
- Version-specific issues with android-sdk
- Compatibility concerns between android-sdk and other technologies
- Debugging strategies for android-sdk applications

### Frequent Mistakes:

- Not handling configuration changes properly, leading to data loss or unexpected behavior.
- Performing long-running operations on the main thread, causing the UI to freeze.
- Leaking `Context` objects, leading to memory leaks.
- Not properly validating user input, leading to security vulnerabilities.
- Not handling exceptions properly, leading to app crashes.
- Overusing `AsyncTask` and not managing thread pools correctly.

### Edge Cases:

- Handling network connectivity issues gracefully.
- Managing different screen sizes and densities.
- Dealing with low-memory conditions.
- Handling background tasks properly to avoid ANRs (Application Not Responding).
- Handling different Android versions and API levels.

### Version-Specific Issues:

- Be aware of API level differences and use `@TargetApi` and `@RequiresApi` annotations to handle them correctly.
- Test your app on different Android versions to ensure compatibility.

### Compatibility Concerns:

- Ensure compatibility with different device manufacturers and hardware configurations.
- Test your app on different screen sizes and densities.

### Debugging Strategies:

- Use the Android Debug Bridge (ADB) to connect to devices and emulators for debugging.
- Use the Android Profiler to analyze app performance and identify bottlenecks.
- Use logging statements to track the flow of execution and identify errors.
- Use the debugger to step through the code and inspect variables.

## Tooling and Environment

### Recommended Tools:

- Android Studio: The official IDE for Android development.
- Gradle: The build system for Android projects.
- ADB (Android Debug Bridge): A command-line tool for communicating with Android devices and emulators.
- Android Profiler: A tool for analyzing app performance.
- LeakCanary: A library for detecting memory leaks.
- Timber: A logging library for Android.

### Build Configuration:

- MUST use gradle with gradle wrapper to build Android library or Java library
- MUST have consistent minSdk and supportLib versions with other SDKs
- SHOULD use shared build configuration for consistent version with other SDKs/libraries
- MAY use manual configuration with values (same as in shared build configuration), if so SHOULD use a recent minSdk version unless there is an engineering or product decision to support older versions
- SHOULD use latest versions of Android X libraries (if required)

- Configure build variants for different environments (debug, release, staging).
- Use ProGuard or R8 to shrink, obfuscate, and optimize the code for release builds.
- Configure signing configurations to sign the APK for distribution.

### Linting and Formatting:

- Enforce code style guidelines using lint and code formatters (e.g., ktlint for Kotlin).
- Configure lint to run automatically during the build process.
- Resolve lint warnings and errors to improve code quality.

### Deployment:

- Publish libraries to a Maven repository (Maven Central or a private repository).
- Deploy apps to the Google Play Store.
- Implement beta testing using Google Play Console.

### CI/CD Integration:

- Integrate the build process with a CI/CD system (e.g., Jenkins, CircleCI, GitHub Actions).
- Automate testing, linting, and code formatting as part of the CI/CD pipeline.
- Automate deployment to the Google Play Store.

## Additional Considerations
- Limit impact on consumers. SHOULD limit dependencies as much as possible primarily due to risk of version clashes
- RECOMMENDED not to have any exported dependency on other SDK modules e.g. if your module needs authentication, do not depend on Authentication SDK. Instead, require your library consumer to pass an access token string to your API instead.
- Expose minimal Surface area. Keep everything private unless it needs to be public. Every public class, method, property & resource MUST have a strong reason for being public. MAY keep internal source code in a /Private folder to allow tools to more easily ignore those files
- MUST hide classes from generated API documentation that are required to be public solely for technical reasons and are not intended for SDK consumers
- SHOULD declare classes as final to avoid unexpected high-jacking of SDK behavior by subclasses
- For better testability of your library it is RECOMMENDED to use dependency injection
- MUST follow Semantic Versioning
- MUST tag source control management revision/commit that is published
- RECOMMENDED For Android: Use gradle-versions-plugin for automated versioning by git tag