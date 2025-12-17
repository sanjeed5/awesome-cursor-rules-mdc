---
description: Definitive guidelines for writing high-quality, maintainable, and performant Java code, adhering to modern best practices (Java 21/25) and enterprise standards like Google Java Style.
globs: **/*
---
# Java Best Practices

This document outlines the definitive best practices for Java development within our team. Adherence to these rules ensures consistent, high-quality, and maintainable code, leveraging modern Java features and enterprise standards.

## 1. Code Organization and Structure

### 1.1. Style Guide Adherence
Always follow the [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html) for formatting, naming, and structural conventions. Configure your IDE (e.g., IntelliJ IDEA, VS Code with appropriate plugins) to automatically apply this style.

### 1.2. File Naming and Encoding
Source files must be named after the single top-level class they contain, with a `.java` extension. All source files must use **UTF-8** encoding.

### 1.3. Indentation and Whitespace
Use **4 spaces** for indentation. **Never use tabs.** For wrapped lines, use **8 spaces** (twice the normal indentation).

❌ BAD
```java
public class MyClass {
  void myMethod() {
		// Tab used for indentation
		System.out.println("Hello");
  }
}
```

✅ GOOD
```java
public class MyClass {
    void myMethod() {
        // 4 spaces for indentation
        System.out.println("Hello");
    }
}
```

### 1.4. Braces
Always use K&R style braces for non-empty blocks. Braces are mandatory for `if`, `else`, `for`, `do`, and `while` statements, even for single-line bodies.

❌ BAD
```java
if (condition)
    doSomething(); // Missing braces

while (true){ // Bad K&R style
    doSomething();
}
```

✅ GOOD
```java
if (condition) {
    doSomething();
}

while (true) {
    doSomethingElse();
}
```

### 1.5. Package and Import Statements
Every class must belong to a package. Use explicit imports; **wildcard imports (`*`) are forbidden**. Imports are ordered: static imports, then non-static imports, separated by a single blank line, both groups in ASCII sort order.

❌ BAD
```java
import java.util.*; // Wildcard import
import static com.example.Constants.*; // Wildcard static import

public class MyClass { /* ... */ }
```

✅ GOOD
```java
import static com.example.Constants.DEFAULT_VALUE;

import java.util.ArrayList;
import java.util.List;

public class MyClass { /* ... */ }
```

### 1.6. Naming Conventions
Adhere strictly to these conventions:
*   **Packages**: `com.yourcompany.project.module` (all lowercase, dot-separated).
*   **Classes/Enums/Records/Interfaces**: `PascalCase` (nouns).
*   **Variables/Parameters**: `camelCase`.
*   **Constants**: `SCREAMING_SNAKE_CASE` (for `static final` fields).
*   **Methods**: `camelCase` (verbs).
*   **Boolean Variables/Methods**: Prefix with `is`, `has`, `can`, `should`.
*   **Collections**: Use plural names (e.g., `users`, `messages`).
*   **Test Methods**: `featureUnderTest_testScenario_expectedBehavior()`.

❌ BAD
```java
public class userManager { // Class name not PascalCase
    public static final String default_user_name = "guest"; // Constant not SCREAMING_SNAKE_CASE
    private List userList; // Collection not plural
    public boolean checkUser(User u) { /* ... */ } // Boolean method name unclear
}
```

✅ GOOD
```java
package com.example.app.users;

import java.util.List;

public class UserManager {
    public static final String DEFAULT_USER_NAME = "guest";
    private List<User> users;

    public boolean isValidUser(User user) { /* ... */ }
}
```

## 2. Common Patterns and Anti-patterns

### 2.1. Prefer Immutability
Design classes to be immutable whenever possible. This simplifies concurrency and reasoning about state. Use `record` classes (Java 16+) for simple immutable data carriers.

❌ BAD
```java
public class User {
    private String name;
    public User(String name) { this.name = name; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; } // Mutable setter
}
```

✅ GOOD
```java
public record User(String name) {} // Immutable record (Java 16+)
// Or traditional immutable class:
public final class ImmutableUser {
    private final String name;
    public ImmutableUser(String name) { this.name = name; }
    public String getName() { return name; }
}
```

### 2.2. Embrace Functional Programming with Streams and Optionals
Leverage Java's Stream API for collection processing and `Optional` for handling potentially absent values.

❌ BAD
```java
List<String> names = getUsers().stream()
    .filter(u -> u.getAge() > 18)
    .map(User::getName)
    .collect(Collectors.toList());

// Handling null:
User user = findUserById(123L);
if (user != null) {
    System.out.println(user.getName());
} else {
    System.out.println("User not found");
}
```

✅ GOOD
```java
List<String> names = users.stream()
    .filter(user -> user.getAge() > 18)
    .map(User::getName)
    .toList(); // Java 16+ for .toList()

// Handling Optional:
findUserById(123L)
    .map(User::getName)
    .ifPresentOrElse(
        System.out::println,
        () -> System.out.println("User not found")
    );
```

### 2.3. Resource Management with Try-with-resources
Always use try-with-resources for any resource that implements `AutoCloseable`.

❌ BAD
```java
FileInputStream fis = null;
try {
    fis = new FileInputStream("file.txt");
    // ... read from fis
} catch (IOException e) {
    e.printStackTrace();
} finally {
    if (fis != null) {
        try {
            fis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

✅ GOOD
```java
try (FileInputStream fis = new FileInputStream("file.txt")) {
    // ... read from fis
} catch (IOException e) {
    System.err.println("Error reading file: " + e.getMessage());
}
```

## 3. Performance Considerations

### 3.1. Efficient String Concatenation
Use `StringBuilder` for concatenating multiple strings in loops or when building complex strings.

❌ BAD
```java
String result = "";
for (int i = 0; i < 1000; i++) {
    result += i; // Creates many intermediate String objects
}
```

✅ GOOD
```java
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.append(i);
}
String result = sb.toString();
```

### 3.2. Logging Performance
Avoid expensive string formatting or method calls in logging statements unless the log level is enabled. Use parameterized logging.

❌ BAD
```java
logger.debug("Processing user: " + user.getName() + " with ID: " + user.getId()); // String concatenation always happens
```

✅ GOOD
```java
logger.debug("Processing user: {} with ID: {}", user.getName(), user.getId()); // Parameters only evaluated if debug is enabled
```

## 4. Common Pitfalls and Gotchas

### 4.1. Avoid Mutable Static State
Mutable static fields are a common source of concurrency bugs and make testing difficult. Minimize their use.

❌ BAD
```java
public class Counter {
    public static int count = 0; // Mutable static field
    public static void increment() { count++; }
}
```

✅ GOOD
```java
// Use dependency injection for stateful services or pass state explicitly
public class CounterService {
    private int count = 0; // Instance field
    public synchronized void increment() { count++; } // Thread-safe instance method
    public int getCount() { return count; }
}
```

### 4.2. Dependency Version Management
Always pin dependency versions. Use Bill-of-Materials (BOMs) for managing compatible versions across related libraries (e.g., Spring Boot BOM, Google Cloud Libraries BOM).

❌ BAD (Maven)
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
    <!-- No version specified, relies on parent or latest -->
</dependency>
```

✅ GOOD (Maven with BOM)
```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-dependencies</artifactId>
            <version>3.2.5</version> <!-- Pin BOM version -->
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
        <!-- Version inherited from BOM -->
    </dependency>
</dependencies>
```

### 4.3. Correct `equals()` and `hashCode()` Implementation
Always override both `equals()` and `hashCode()` together, or neither. Use `Objects.equals()` and `Objects.hash()` for robust implementations. For data classes, `record` types handle this automatically.

❌ BAD
```java
public class Point {
    private int x, y;
    // Only equals() overridden, or incorrectly
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Point point = (Point) o;
        return x == point.x && y == point.y;
    }
    // Missing hashCode()
}
```

✅ GOOD
```java
import java.util.Objects;

public final class Point { // Make final if not intended for extension
    private final int x, y;
    public Point(int x, int y) { this.x = x; this.y = y; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Point point = (Point) o;
        return