---
description: Enforce modern Gradle build practices for Java and Android projects, focusing on Kotlin DSL, modularity, version catalogs, and performance optimizations.
globs: **/*.java
---
# Gradle Best Practices

This guide outlines the definitive best practices for building robust, maintainable, and performant Java and Android projects with Gradle, leveraging the Kotlin DSL and modern features.

## 1. Adopt Kotlin DSL for Build Scripts

Always use the Kotlin DSL (`build.gradle.kts`) for new projects and when refactoring existing Groovy DSL (`build.gradle`) scripts. It offers strict typing, superior IDE support, and a unified language stack for Kotlin projects.

❌ BAD: Groovy DSL
```groovy
// build.gradle
plugins {
    id 'java-library'
}
dependencies {
    implementation 'com.google.guava:guava:31.1-jre'
}
```

✅ GOOD: Kotlin DSL
```kotlin
// build.gradle.kts
plugins {
    `java-library`
}
dependencies {
    implementation("com.google.guava:guava:31.1-jre")
}
```

## 2. Modularize Your Build Early and Aggressively

Split your codebase into small, logical Gradle projects (modules) from the start. This is critical for leveraging Gradle's work avoidance, parallel execution, and minimizing compilation classpaths.

❌ BAD: Monolithic `app` module
```
├── app
│   ├── build.gradle.kts
│   └── src/main/java/org/example/... (all code here)
└── settings.gradle.kts
    include("app")
```

✅ GOOD: Modularized structure
```
├── app
│   └── build.gradle.kts
├── features/feature-a
│   └── build.gradle.kts
├── core/domain
│   └── build.gradle.kts
└── settings.gradle.kts
    include("app", "features:feature-a", "core:domain")
```
*   **Guideline**: Each module should have a clear, single responsibility (e.g., `api`, `implementation`, `feature`, `ui`, `core`).

## 3. Centralize Dependencies with Version Catalogs

Manage all library versions in a single `gradle/libs.versions.toml` file. This eliminates duplication, simplifies upgrades, and ensures consistency across modules.

❌ BAD: Scattered versions or `ext` properties
```kotlin
// build.gradle.kts (multiple files)
val guavaVersion = "31.1-jre" // Or hardcoded
dependencies {
    implementation("com.google.guava:guava:$guavaVersion")
}
```

✅ GOOD: `gradle/libs.versions.toml`
```toml
# gradle/libs.versions.toml
[versions]
guava = "31.1-jre"
[libraries]
guava = { module = "com.google.guava:guava", version.ref = "guava" }
```
```kotlin
// build.gradle.kts
dependencies {
    implementation(libs.guava)
}
```
*   **Naming**: Use descriptive, hyphen-separated names in `libs.versions.toml` (e.g., `junit-jupiter`, `spring-boot-starter-web`). Convert internal dashes to camelCase for access in build scripts (e.g., `libs.spring.boot.starter.web`).

## 4. Apply Plugins Using the `plugins {}` Block

Always use the declarative `plugins {}` block for applying plugins. This is more concise, less error-prone, and allows Gradle to optimize plugin loading.

❌ BAD: Legacy `apply` statements
```groovy
// build.gradle
buildscript {
    repositories { gradlePluginPortal() }
    dependencies { classpath("com.google.protobuf:com.google.protobuf.gradle.plugin:0.9.4") }
}
apply plugin: "com.google.protobuf"
```

✅ GOOD: `plugins {}` block
```kotlin
// build.gradle.kts
plugins {
    id("java-library")
    id("com.google.protobuf").version("0.9.4")
}
```
*   **Note**: For plugins defined in your Version Catalog, use `alias(libs.plugins.myPlugin)`.

## 5. Leverage the Configuration Cache

Enable the Configuration Cache to dramatically speed up build times, especially for large projects. This feature is moving towards default enablement in future Gradle versions.

```bash
# Enable in gradle.properties (root project)
org.gradle.configuration-cache=true
```
*   **Pitfall**: Ensure your build logic is compatible. Avoid global mutable state, direct file system access during configuration, and non-serializable objects. Debug with `./gradlew --configuration-cache-problems`.

## 6. Optimize CI Builds with Gradle Wrapper and Caching

Use the Gradle Wrapper (`./gradlew`) for all CI builds to ensure consistent Gradle versions. Cache the Gradle user home directory (or use `gradle/actions/setup-gradle`) to speed up dependency resolution and build artifact caching.

```yaml
# .github/workflows/ci.yml (GitHub Actions example)
steps:
  - uses: actions/checkout@v4
  - name: Set up JDK 17
    uses: actions/setup-java@v4
    with:
      java-version: '17'
      distribution: 'temurin'
  - name: Setup Gradle
    uses: gradle/actions/setup-gradle@v3 # Caches automatically
  - name: Build with Gradle
    run: ./gradlew build
```

## 7. Avoid Internal Gradle APIs

Never use APIs from packages containing `internal` or types with `Internal`/`Impl` suffixes. These APIs are unstable and subject to breaking changes without notice, causing build failures during Gradle upgrades.

❌ BAD: Using `AttributeContainerInternal`
```kotlin
// build.gradle.kts
import org.gradle.api.internal.attributes.AttributeContainerInternal // ❌ DANGER
configurations.create("bad") {
    val badMap = (attributes as AttributeContainerInternal).asMap()
    // ...
}
```

✅ GOOD: Stick to public APIs
*   **Guideline**: If a public API is missing functionality, submit a feature request to Gradle.

## 8. Maintain Latest Minor Gradle Version and Plugins

Regularly update Gradle to the latest minor version of your targeted major release. Update plugins to their latest compatible versions. This ensures access to performance improvements, bug fixes, and security patches.

```bash
# Update Gradle Wrapper
./gradlew wrapper --gradle-version <latest-version> --distribution-type all
```
*   **Strategy**: Upgrade Gradle first, then plugins. Consult changelogs for compatibility.

## 9. Centralize Repository Declarations in `settings.gradle.kts`

Declare all dependency repositories in `settings.gradle.kts` to ensure consistency across all projects and subprojects.

❌ BAD: Repositories in `build.gradle.kts`
```kotlin
// app/build.gradle.kts
repositories {
    mavenCentral()
    google()
}
```

✅ GOOD: Repositories in `settings.gradle.kts`
```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS) // Enforce
    repositories {
        mavenCentral()
        google()
    }
}
```
*   **Note**: `repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)` prevents subprojects from declaring their own repositories.

## 10. Use `java-library` Plugin for Library Modules

For modules that produce a library (JAR) to be consumed by other modules, use the `java-library` plugin. It provides `api` and `implementation` configurations, promoting better encapsulation and faster compilation.

```kotlin
// core/domain/build.gradle.kts
plugins {
    `java-library`
}
dependencies {
    api("org.slf4j:slf4j-api:2.0.13") // Public API
    implementation("com.fasterxml.jackson.core:jackson-databind:2.17.1") // Internal detail
}
```
*   **Guideline**: Use `api` for dependencies that are part of your module's public API. Use `implementation` for internal dependencies.

## 11. Do Not Put Source Files in the Root Project

The root project should only contain build configuration (e.g., `settings.gradle.kts`, `build.gradle.kts` for global config, `gradle.properties`). All application source code belongs in subprojects.

❌ BAD: Source in root
```
├── build.gradle.kts
├── src/main/java/org/example/RootApp.java // ❌
└── settings.gradle.kts
```

✅ GOOD: Source in dedicated subproject
```
├── build.gradle.kts
├── app/build.gradle.kts
├── app/src/main/java/org/example/App.java // ✅
└── settings.gradle.kts
    include("app")
```