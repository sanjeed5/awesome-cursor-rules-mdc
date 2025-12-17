---
description: This guide provides opinionated, actionable best practices for Maven projects, focusing on reproducible builds, efficient dependency management, and high code quality.
globs: **/*.java
---
# Maven Best Practices

Maven is the definitive build tool for modern Java. Adhere to these rules for consistent, maintainable, and high-quality projects.

## Code Organization and Structure

### Centralize Configuration with a Parent POM/BOM

Always use a parent POM (or Bill of Materials - BOM) to define shared properties, plugin versions, and a `<dependencyManagement>` section. This guarantees consistent library versions and build behavior across all modules.

❌ BAD: Duplicating dependency versions and plugin configurations in every child `pom.xml`.
```xml
<!-- child-module/pom.xml -->
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
        <version>3.2.0</version> <!-- Hardcoded, duplicated -->
    </dependency>
</dependencies>
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.11.0</version> <!-- Hardcoded, duplicated -->
            <configuration><release>17</release></configuration>
        </plugin>
    </plugins>
</build>
```

✅ GOOD: Define a parent POM with `<dependencyManagement>` and `<pluginManagement>`. Child modules inherit these.

```xml
<!-- parent/pom.xml -->
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.mycompany</groupId>
    <artifactId>my-parent</artifactId>
    <version>1.0.0-SNAPSHOT</version>
    <packaging>pom</packaging>
    <properties>
        <java.version>17</java.version>
        <spring-boot.version>3.2.0</spring-boot.version>
        <maven.compiler.plugin.version>3.11.0</maven.compiler.plugin.version>
    </properties>
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-dependencies</artifactId>
                <version>${spring-boot.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
    <build>
        <pluginManagement>
            <plugins>
                <plugin>
                    <groupId>org.apache.maven.plugins</groupId>
                    <artifactId>maven-compiler-plugin</artifactId>
                    <version>${maven.compiler.plugin.version}</version>
                    <configuration><release>${java.version}</release></configuration>
                </plugin>
            </plugins>
        </pluginManagement>
    </build>
</project>

<!-- child-module/pom.xml -->
<project>
    <parent>
        <groupId>com.mycompany</groupId>
        <artifactId>my-parent</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>
    <modelVersion>4.0.0</modelVersion>
    <artifactId>my-child-module</artifactId>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <!-- Version inherited from parent's dependencyManagement -->
        </dependency>
    </dependencies>
</project>
```

## Common Patterns and Anti-patterns

### Declare the Narrowest Possible Dependency Scope

Limit dependency transitivity and classpath bloat by using the most restrictive scope (`compile`, `provided`, `runtime`, `test`).

❌ BAD: Using `compile` scope for everything, including test-only dependencies.
```xml
<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter-api</artifactId>
    <version>5.11.0</version>
    <!-- Missing scope, defaults to compile -->
</dependency>
```

✅ GOOD: Explicitly define scopes.
*   `compile` (default): Available in all classpaths, propagated to dependents. For core libraries.
*   `provided`: Expected to be provided by JDK or container at runtime. For Servlet API, Java EE APIs.
*   `runtime`: Not needed for compilation, but for execution. For JDBC drivers.
*   `test`: Only for test compilation and execution. For JUnit, Mockito.

```xml
<dependencies>
    <dependency>
        <groupId>org.junit.jupiter</groupId>
        <artifactId>junit-jupiter-api</artifactId>
        <scope>test</scope> <!-- Correct scope for testing -->
    </dependency>
    <dependency>
        <groupId>jakarta.servlet</groupId>
        <artifactId>jakarta.servlet-api</artifactId>
        <version>6.0.0</version>
        <scope>provided</scope> <!-- Provided by web container -->
    </dependency>
</dependencies>
```

### Manage Transitive Dependencies with Exclusions

Prevent unwanted or conflicting transitive dependencies from polluting your classpath.

❌ BAD: Relying on Maven's "nearest definition" for conflicting transitive dependencies, leading to unpredictable builds or runtime issues.
```xml
<!-- Project A depends on B (which depends on D 1.0) and C (which depends on D 2.0) -->
<!-- Maven might pick D 1.0 or D 2.0 based on declaration order/depth -->
<dependency>
    <groupId>com.example</groupId>
    <artifactId>library-b</artifactId>
    <version>1.0</version>
</dependency>
<dependency>
    <groupId>com.example</groupId>
    <artifactId>library-c</artifactId>
    <version>1.0</version>
</dependency>
```

✅ GOOD: Explicitly exclude problematic transitive dependencies and declare the desired version directly.

```xml
<dependencies>
    <dependency>
        <groupId>com.example</groupId>
        <artifactId>library-b</artifactId>
        <version>1.0</version>
        <exclusions>
            <exclusion>
                <groupId>com.example</groupId>
                <artifactId>dependency-d</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
    <dependency>
        <groupId>com.example</groupId>
        <artifactId>library-c</artifactId>
        <version>1.0</version>
    </dependency>
    <dependency>
        <groupId>com.example</groupId>
        <artifactId>dependency-d</artifactId>
        <version>2.0</version> <!-- Explicitly declare the desired version -->
    </dependency>
</dependencies>
```

## Performance Considerations

### Use Maven Wrapper for Consistent Builds

Ensure all developers and CI/CD pipelines use the exact same Maven version, preventing "works on my machine" issues related to Maven itself.

✅ GOOD: Include Maven Wrapper in your project.
```bash
mvn wrapper:wrapper
```
Then, use `./mvnw` (or `mvnw.cmd` on Windows) instead of `mvn`.
```bash
./mvnw clean install
```

## Common Pitfalls and Gotchas

### Avoid Over-reliance on Maven Profiles for Environment Configuration

Modern applications should be environment-agnostic (12-factor app principles). Use external configuration (environment variables, configuration servers) for environment-specific settings, not Maven profiles.

❌ BAD: Using profiles to inject environment-specific database URLs or API keys.
```xml
<!-- profile in pom.xml -->
<profiles>
    <profile>
        <id>dev</id>
        <properties>
            <db.url>jdbc:h2:mem:devdb</db.url>
        </properties>
    </profile>
    <profile>
        <id>prod</id>
        <properties>
            <db.url>jdbc:postgresql://prod-db:5432/appdb</db.url>
        </properties>
    </profile>
</profiles>
```

✅ GOOD: Keep application configuration external to the build. Maven profiles are acceptable for *build process* variations (e.g., different test suites, packaging types), but not runtime environment configuration.

## Testing Approaches

### Integrate Static Analysis Plugins

Enforce coding standards and identify potential issues early by integrating static analysis tools like Checkstyle, SpotBugs, or PMD into the Maven build lifecycle (e.g., `verify` phase).

✅ GOOD: Configure static analysis plugins in your parent POM's `<build>` section.

```xml
<!-- parent/pom.xml -->
<build>
    <pluginManagement>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-checkstyle-plugin</artifactId>
                <version>3.3.1</version>
                <executions>
                    <execution>
                        <id>validate</id>
                        <phase>validate</phase>
                        <goals>
                            <goal>check</goal>
                        </goals>
                        <configuration>
                            <configLocation>checkstyle.xml</configLocation>
                            <failsOnError>true</failsOnError>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
            <plugin>
                <groupId>com.github.spotbugs</groupId>
                <artifactId>spotbugs-maven-plugin</artifactId>
                <version>4.8.3.0</version>
                <executions>
                    <execution>
                        <id>analyze</id>
                        <phase>verify</phase>
                        <goals>
                            <goal>check</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </pluginManagement>
</build>
```