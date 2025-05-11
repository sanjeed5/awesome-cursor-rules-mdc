---
description: Comprehensive rules for Gradle best practices, covering code organization, performance, security, testing, and more. Provides actionable guidance to improve Gradle project maintainability and efficiency.
globs: *.gradle*
---
- **Use the Gradle Wrapper**: Always use the Gradle wrapper (`gradlew` and `gradlew.bat`) to ensure consistent Gradle versions across different environments. This avoids compatibility issues and simplifies collaboration.
  - Command: `gradle wrapper` to generate the wrapper files.

- **Organize Build Files**: Maintain a clear structure by separating project-level (`build.gradle` in the root) and module-level (`build.gradle` in each module) build files. Use `settings.gradle` (or `settings.gradle.kts` for Kotlin DSL) to define included modules.
  - Improves performance by avoiding unnecessary directory searches.
  - Ensures proper project inclusion in multi-module builds.

- **Optimize Build Performance**: Employ several strategies to improve build speed:
  - **Enable parallel builds**: Add `org.gradle.parallel=true` to `gradle.properties`.
  - **Use build caching**: Add `org.gradle.caching=true` to `gradle.properties`.
  - **Configure daemon**: Add `org.gradle.daemon=true` to `gradle.properties` (typically enabled by default).
  - **Minimize dynamic dependencies**: Use fixed versions instead of ranges or `latest.release`.
  - **Configure JVM Memory**: Set appropriate JVM memory in `gradle.properties` like `org.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=1g`
  - **Incremental build**: Avoid `gradle clean` unless absolutely necessary to leverage incremental build feature.
  - **Configure task inputs and outputs**:  Declare task inputs and outputs properly to enable up-to-date checks and caching.

- **Security Practices**: Protect sensitive information by:
  - **Storing credentials in `gradle.properties`**: Exclude `gradle.properties` from version control (add it to `.gitignore`).
  - **Using environment variables**: Access credentials from environment variables during the build process for CI/CD environments.
  - **Avoiding hardcoding secrets**: Never hardcode API keys, passwords, or other sensitive data in build files or source code.

- **Automate Code Standards**: Implement custom Gradle tasks or plugins to enforce coding standards:
  - **Use linting tools**: Integrate linters (e.g., Checkstyle, SpotBugs, PMD) into the build process.
  - **Enforce code formatting**: Use code formatters (e.g., ktlint for Kotlin) and enforce formatting rules.
  - **Create custom tasks**: Define custom tasks to check for specific code patterns or enforce project-specific rules.

- **Separate Source Files**: Organize source files by language (e.g., `src/main/java`, `src/main/kotlin`) and test type (e.g., `src/test/java`, `src/integrationTest/java`).
  - Improves readability and maintainability.
  - Enables independent execution of different test types.

- **Use Standard Conventions**: Stick to Gradle's default conventions as much as possible.
  - Simplifies project structure and reduces configuration overhead.
  - Makes it easier for new developers to understand the project.

- **Always Define a Settings File**: Include `settings.gradle` (or `settings.gradle.kts`) in the root directory to avoid performance impacts during project discovery.
  - Explicitly defines the project name and included subprojects.

- **Use `buildSrc` to Abstract Imperative Logic**: Encapsulate complex build logic in the `buildSrc` directory.
  - `buildSrc` is treated as an included build, automatically compiled and added to the build script classpath.
  - Provides a clean separation of concerns and improves code reusability.
  - Easier to maintain, refactor, and test the code in `buildSrc`.

- **Declare Properties in `gradle.properties` File**: Store build configuration properties in `gradle.properties` rather than in build scripts.
  - Improves build script readability and maintainability.
  - Allows for easy modification of build properties without changing the build script.

- **Avoid Overlapping Task Outputs**: Ensure that tasks write outputs to unique directories to prevent conflicts and ensure proper up-to-date checking.
  - Prevents Gradle's build cache from being compromised.

- **Standardize Builds with a Custom Gradle Distribution**: Create a custom Gradle distribution with initialization scripts to enforce common conventions and rules across all projects in an organization.
  - Ensures consistent build environments and simplifies maintenance.

- **Stop Cleaning Your Project (Unless Necessary)**: Avoid running `gradle clean` unnecessarily, as it prevents Gradle from using incremental build features and significantly slows down the build process.

- **Move Tasks to `buildSrc`**: Move custom task definitions to the `buildSrc` directory to keep build scripts clean and reusable.

- **Run Tests in Parallel**: Enable parallel test execution to reduce test execution time:
  - Add `maxParallelForks = <number_of_cores>` to the `test` task configuration in `build.gradle`.

- **Version Your Project**: Use a versioning scheme (e.g., semantic versioning) to track changes and releases.
  - Use plugins like `axion-release` to automate versioning based on Git tags.

- **Encapsulate Task Declarations in a Plugin**: Move task declarations into custom Gradle plugins to promote reusability and reduce duplication.

- **Use the Latest Gradle Version**: Keep Gradle updated to benefit from performance improvements, new features, and security patches.

- **Optimize Your Repositories**: Declare repositories in the correct order to avoid unnecessary network requests.
  - List frequently accessed repositories first.

- **Never Commit Passwords**: Externalize credentials using `gradle.properties` or environment variables.

- **Adopt Kotlin DSL**: Favor Kotlin DSL (`build.gradle.kts`) over Groovy DSL (`build.gradle`). Kotlin DSL provides type safety, better IDE support, and improved code completion.

- **Dependency Management**: 
   - **Use dependency catalogs**: Define dependencies in a central `libs.versions.toml` file for consistent dependency management across modules (available with Gradle 7.4+).
   - **Avoid dynamic versions**: Always use explicit dependency versions to ensure reproducible builds.
   - **Check for dependency updates**: Use dependency analysis tools to identify outdated dependencies and security vulnerabilities.

- **Code Organization and Structure (Expanded)**:
   - **Directory Structure**:  Adhere to standard directory structures, such as `src/main/java` for Java source code, `src/main/resources` for resources, `src/test/java` for unit tests, and `src/integrationTest/java` for integration tests.
   - **File Naming**: Follow consistent naming conventions for Gradle files (e.g., `build.gradle`, `settings.gradle`, `gradle.properties`).
   - **Module Organization**: Organize projects into modules based on functionality or feature sets.  This promotes code reuse, improves build times, and allows for more granular dependency management.
   - **Component Architecture**:  Consider using component-based architectures to isolate independent parts of your code.  Each component has its own build file, source folder, and dependencies. This promotes modularity and reusability.
   - **Code Splitting**: Decompose complex build scripts and tasks into smaller, more manageable pieces.

- **Common Patterns and Anti-patterns (Expanded)**:
   - **Dependency Injection**: Leverage dependency injection frameworks (e.g., Dagger, Guice) to manage dependencies in custom tasks and plugins.
   - **Declarative Configuration**: Prefer declarative configuration over imperative scripting. This promotes readability and simplifies maintenance.
   - **Avoid Global State**:  Minimize the use of global state in custom tasks and plugins to prevent unintended side effects.
   - **Tight Coupling**:  Avoid tight coupling between build logic and application code. This promotes modularity and testability.

- **Performance Considerations (Expanded)**:
   - **Configuration Avoidance**: Use `afterEvaluate` sparingly, as it can delay configuration and impact build performance.  Consider using configuration caching instead.
   - **Task Ordering**: Ensure tasks are executed in the correct order. Inappropriate task dependencies slow down the build process.

- **Testing Approaches (Expanded)**:
   - **Test Organization**: Organize tests in a clear and consistent manner. Follow a similar directory structure as source code (e.g., `src/test/java`).
   - **Test Doubles**: Use mocking and stubbing techniques to isolate units of code for testing. Libraries like Mockito and PowerMock facilitate test double creation.  Use carefully PowerMock, as can be problematic.

- **Tooling and Environment (Expanded)**:
   - **Android Studio**: Use Android Studio as the primary IDE for Android projects. It offers excellent Gradle integration and debugging support.
   - **CI/CD Integration**: Integrate Gradle builds with CI/CD systems (e.g., Jenkins, GitLab CI, GitHub Actions) to automate testing and deployment processes.
   - **Code Analysis Tools**: Integrate code analysis tools (e.g., SonarQube) into the build process to identify code quality issues and security vulnerabilities.
   - **IDE plugins**: Use IDE plugins, such as the Gradle build scan plugin, to analyze and optimize build performance.

- **Common Pitfalls and Gotchas (Expanded)**:
  - **Dependency Conflicts**: Be aware of potential dependency conflicts and use dependency resolution strategies to manage them.
  - **Cache Invalidation**: Understand how Gradle's build cache works and how to invalidate it when necessary.
  - **Plugin Compatibility**: Ensure compatibility between Gradle versions and plugins.  Upgrade plugins carefully.
  - **Daemon issues**: The gradle daemon sometime cause unexplainable build errors. Restarting or stopping it can solve some build issues.

- **Android Specific Best Practices**:
  - **Resource Management**: Optimize image resources to reduce APK size. Use vector drawables where possible.
  - **Proguard/R8**: Use Proguard (or R8) to shrink and obfuscate code in release builds.
  - **APK Analyzer**: Use APK Analyzer to inspect the contents of APK files and identify areas for optimization.
  - **Build variants**: Use build variants to create different versions of the app for different purposes (e.g., debug, release, flavor-specific).

- **Multi-project Builds**:
   - **Centralized Dependency Management:**  Manage dependencies centrally within the root project or through a convention plugin, applying these dependencies to child modules.  This promotes consistency and simplifies dependency updates.
   - **Convention Plugins:** Create custom plugins that apply common configurations (dependencies, plugins, settings) to all subprojects. This reduces code duplication and improves maintainability.
   - **Composite Builds:** Use composite builds (including local projects in your build) for collaborative development or when working on multiple projects simultaneously.

- **Kotlin DSL Specific Best Practices**:
  - **Explicit Typing**: Use explicit typing to leverage Kotlin's type safety and improve code readability.
  - **Extension Functions**: Define extension functions to extend the Gradle API and add custom functionality.
  - **Coroutines:**  Utilize Kotlin coroutines for asynchronous operations in custom tasks and plugins.

- **Monitoring and Observability**:
  - **Gradle Build Scans:**  Use Gradle Build Scans (using `--scan`) to capture detailed build information, analyze performance bottlenecks, and identify areas for improvement.
  - **Custom Metrics:** Implement custom metrics to track key build statistics (build duration, task execution times) and monitor build health.

- **Continuous Improvement:**
   - **Regular Audits:**  Perform regular audits of Gradle build configurations and scripts to identify areas for optimization and improvement.
   - **Stay Updated:**  Keep abreast of the latest Gradle releases, best practices, and community recommendations.
   - **Share Knowledge:**  Document Gradle build configurations, custom tasks, and plugins. Share knowledge and best practices with other team members.

By adhering to these best practices, developers can create robust, maintainable, and efficient Android applications using Gradle as their build tool.