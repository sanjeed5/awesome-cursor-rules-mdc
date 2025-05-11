---
description: Comprehensive coding standards and best practices for Godot Engine development, covering code organization, performance, testing, and security to ensure maintainable, efficient, and secure game projects. These rules are primarily for GDScript but also reference relevant C# practices where applicable.
globs: *.gd
---
# Godot Engine Best Practices and Coding Standards

This document outlines best practices and coding standards for Godot Engine development using GDScript and C#. Adhering to these guidelines will lead to more maintainable, efficient, and secure game projects.

## 1. Code Organization and Structure

### 1.1. Directory Structure

*   **`scenes/`**: Stores all `.tscn` (scene) and `.escn` (external scene) files. Organize scenes into subdirectories based on game areas, features, or entity types (e.g., `scenes/characters/`, `scenes/levels/`, `scenes/ui/`).
*   **`scripts/`**: Contains all GDScript (`.gd`) and C# (`.cs`) scripts. Mirror the scene directory structure where applicable to maintain consistency (e.g., `scripts/characters/`, `scripts/levels/`).
*   **`assets/` (or `art/` / `textures/` / `audio/`):** Stores all non-code assets like textures, audio files, models, animations, and fonts. Subdivide further based on asset type (e.g., `assets/textures/`, `assets/audio/`, `assets/models/`).
*   **`addons/`**: Reserved for third-party addons and plugins.
*   **`data/` (or `resources/`):** Holds data files like JSON, CSV, or custom resource files used for game configuration, localization, or level data.
*   **`shaders/`**: Stores custom shader files (`.shader`).
*   **`autoload/`**: If using autoloaded scripts (singletons), consider a dedicated folder.


my_project/
├── scenes/
│   ├── characters/
│   │   ├── player.tscn
│   │   └── enemy.tscn
│   ├── levels/
│   │   ├── level_1.tscn
│   │   └── level_2.tscn
│   └── ui/
│       └── main_menu.tscn
├── scripts/
│   ├── characters/
│   │   ├── player.gd
│   │   └── enemy.gd
│   └── ui/
│       └── main_menu.gd
├── assets/
│   ├── textures/
│   │   ├── player.png
│   │   └── enemy.png
│   ├── audio/
│   │   ├── background.ogg
│   │   └── jump.wav
│   └── models/
│       └── character.glb
├── addons/
│   └── ...
├── data/
│   └── levels.json
└── shaders/
    └── water.shader


### 1.2. File Naming Conventions

*   **Scenes:** Use PascalCase (e.g., `Player.tscn`, `MainMenu.tscn`).
*   **Scripts (GDScript):** Use snake_case (e.g., `player_controller.gd`, `ui_manager.gd`).
*   **Scripts (C#):** Use PascalCase (e.g., `PlayerController.cs`, `UIManager.cs`).
*   **Assets:** Use descriptive names with appropriate extensions (e.g., `player_idle.png`, `background_music.ogg`, `sword.glb`). Consider using snake_case for assets as well to maintain consistency.

### 1.3. Module Organization

*   **Separate Concerns:** Divide your project into logical modules based on functionality (e.g., player controller, AI, UI, physics, networking). Each module should have its own directory and scripts.
*   **Autoloads (Singletons):** Use autoloads sparingly. Overuse can lead to tightly coupled code.  Good candidates are global managers like `GameManager`, `InputManager`, or `AudioManager`.
*   **Custom Resources:** Create custom resource types for reusable data containers. This promotes data-driven design (e.g., `CharacterData`, `WeaponData`).

### 1.4. Component Architecture

*   **Favor Composition over Inheritance:** Use nodes and scripts as components that can be attached to other nodes. This allows for more flexible and reusable code. Godot's scene system is built around this concept.
*   **Signals:** Use signals for communication between nodes. This promotes loose coupling and allows nodes to react to events without knowing the details of other nodes.

### 1.5. Code Splitting Strategies

*   **Separate Logic from Presentation:** Keep scene files focused on visual representation and node hierarchy. Move game logic, input handling, and AI to separate scripts.
*   **Helper Functions:** Create utility scripts with reusable functions to avoid code duplication.
*   **Script Classes (GDScript):** Define classes within scripts to encapsulate related data and functions. Use `class_name` to register the class as a global type.
*   **Partial Classes (C#):** Use partial classes to split a large class into multiple files, improving organization.

## 2. Common Patterns and Anti-patterns

### 2.1. Design Patterns

*   **Singleton (Autoload):** As mentioned before, use autoloads for global managers.
*   **Observer (Signals):** Use signals for event-driven communication between nodes.
*   **State Machine:** Implement state machines to manage the behavior of game entities (e.g., player states: idle, walking, jumping, attacking).
*   **Object Pool:** For frequently created and destroyed objects (e.g., bullets, particles), use an object pool to reduce memory allocation overhead.
*   **Factory:** Use factories to create instances of objects based on certain conditions or configurations.

### 2.2. Recommended Approaches for Common Tasks

*   **Input Handling:** Use the `Input` singleton and input actions for consistent input handling. Define input maps in the project settings.
*   **Scene Management:** Use `get_tree().change_scene_to_file()` or `get_tree().change_scene_to_packed()` for scene transitions.
*   **Timers:** Use the `Timer` node for delayed execution or repeated events.  Avoid using `yield(get_tree().create_timer(time), "timeout")` unless absolutely necessary, prefer the Timer node and signals for readability.
*   **Animation:** Utilize the `AnimationPlayer` node for complex animations. For simple animations, consider tweens.
*   **Networking:** Use Godot's built-in networking API for multiplayer games. Consider using a higher-level networking library like ENet or Nakama for more advanced features.

### 2.3. Anti-patterns and Code Smells

*   **God Classes:** Avoid creating large classes that handle too many responsibilities. Break them down into smaller, more focused components.
*   **Tight Coupling:** Minimize dependencies between nodes. Use signals and interfaces to promote loose coupling.
*   **Spaghetti Code:** Avoid deeply nested conditional statements and loops. Use functions and classes to break down complex logic into smaller, more manageable pieces.
*   **Magic Numbers:** Avoid using hardcoded numbers in your code. Use constants or variables instead.
*   **Premature Optimization:** Don't optimize your code before you identify performance bottlenecks. Focus on writing clear and maintainable code first.

### 2.4. State Management

*   **Finite State Machines (FSMs):** Use FSMs to manage the different states of game entities, such as player characters or AI agents. This helps to organize and control complex behavior.
*   **State Design Pattern:** Implement the State design pattern to encapsulate the logic for each state in separate classes or scripts. This makes it easier to add, remove, or modify states without affecting other parts of the code.
*   **Hierarchical State Machines (HSMs):** For more complex state management scenarios, consider using HSMs to create nested states. This allows you to define common behavior in parent states and override it in child states as needed.

### 2.5. Error Handling

*   **`assert()`:** Use `assert()` for debugging to check for conditions that should always be true.  These checks are disabled in release builds.
*   **`try...except` (GDScript) / `try...catch` (C#):** Use try-except/try-catch blocks to handle exceptions and prevent crashes. Handle expected errors gracefully (e.g., file not found).
*   **Error Signals:** Emit custom signals to notify other nodes of errors.  This allows for centralized error handling and logging.
*   **Return Values:** Use return values to indicate success or failure. For example, a function that loads a file could return `OK` or `ERR_FILE_NOT_FOUND`.
*   **Logging:** Use `print()` (for debugging) and a dedicated logging system (for production) to track errors and warnings.  Use a custom logging system that can be disabled in release builds.

## 3. Performance Considerations

### 3.1. Optimization Techniques

*   **Culling:** Use frustum culling and occlusion culling to reduce the number of objects that need to be rendered.
*   **Level of Detail (LOD):** Use LOD to reduce the complexity of models and textures based on their distance from the camera.
*   **Batching:** Batch multiple draw calls into a single draw call to reduce the overhead of rendering.
*   **Object Pooling:** Reuse objects instead of creating and destroying them frequently.
*   **Optimized Data Structures:** Use appropriate data structures for your data (e.g., dictionaries for fast lookups, arrays for ordered data).
*   **Profiling:** Use Godot's built-in profiler to identify performance bottlenecks.
*   **Multi-threading:** Utilize multi-threading for long running operations.
*   **Avoid `get_node()` in Loops:** Cache node references to avoid repeated calls to `get_node()` within loops.
*   **Use Typed Arrays:** In GDScript, use Typed Arrays (e.g., `PackedByteArray`, `PackedFloat32Array`) for efficient storage and manipulation of large amounts of data.

### 3.2. Memory Management

*   **Resource Management:** Load resources only when needed and unload them when they are no longer used. Use `ResourceLoader.load()` and `Resource.free()`.
*   **Circular References:** Avoid circular references between objects, as this can prevent them from being garbage collected.
*   **Object Pooling:** As mentioned before, use object pooling to reduce memory allocation overhead.
*   **Weak References:**  If you need to reference an object without preventing it from being garbage collected, use a `WeakRef`.

### 3.3. Rendering Optimization

*   **Reduce Draw Calls:** Minimize the number of draw calls by using techniques such as batching and instancing.
*   **Optimize Shaders:** Write efficient shaders that use the minimum number of instructions.
*   **Texture Compression:** Use compressed textures to reduce memory usage and improve performance.
*   **Mipmaps:** Use mipmaps to improve texture filtering quality and reduce aliasing.
*   **Limit Overdraw:** Reduce overdraw by avoiding overlapping transparent objects.
*   **CanvasItem Z Index:** Group similar Z index values to improve rendering performance.

### 3.4. Bundle Size Optimization

*   **Compress Textures:** Use compressed textures to reduce the size of your textures.
*   **Optimize Audio Files:** Use compressed audio formats such as OGG Vorbis or MP3.
*   **Remove Unused Assets:** Remove any unused assets from your project.
*   **Use Asset Packs:** Use asset packs to share assets between multiple projects.
*   **Enable Texture Compression:**  In project settings, enable texture compression for release builds.

### 3.5. Lazy Loading

*   **Load Scenes Asynchronously:** Load scenes in the background to prevent the game from freezing during scene transitions.  Use `ResourceLoader.load_interactive()`.
*   **Stream Audio:** Stream audio files instead of loading them into memory all at once.
*   **Load Resources on Demand:** Load resources only when they are needed, such as when a player enters a new area.

## 4. Security Best Practices

### 4.1. Common Vulnerabilities

*   **Injection Attacks:** Prevent SQL injection, code injection, and other injection attacks by validating all user inputs.
*   **Cross-Site Scripting (XSS):**  Not typically a concern for standalone games, but relevant if integrating web-based content or APIs.
*   **Denial of Service (DoS):** Protect your game from DoS attacks by limiting the number of requests that can be made from a single IP address.
*   **Data Tampering:** Prevent players from tampering with game data by encrypting sensitive data and validating it on the server.
*   **Reverse Engineering:**  Obfuscate your code to make it more difficult for players to reverse engineer your game.

### 4.2. Input Validation

*   **Validate All Inputs:** Validate all user inputs, including text fields, number fields, and dropdown menus.
*   **Use Regular Expressions:** Use regular expressions to validate input formats.
*   **Sanitize Inputs:** Sanitize inputs to remove potentially harmful characters.
*   **Limit Input Length:** Limit the length of input fields to prevent buffer overflows.
*   **Client-Side and Server-Side Validation:** Implement both client-side and server-side validation to prevent malicious users from bypassing client-side checks.

### 4.3. Authentication and Authorization

*   **Use Secure Authentication Protocols:** Use secure authentication protocols such as OAuth 2.0 or JWT.
*   **Store Passwords Securely:** Store passwords using a strong hashing algorithm such as Argon2 or bcrypt.
*   **Implement Role-Based Access Control (RBAC):** Use RBAC to control access to different parts of your game based on user roles.
*   **Two-Factor Authentication (2FA):** Implement 2FA to add an extra layer of security to user accounts.

### 4.4. Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data such as passwords, API keys, and user data using a strong encryption algorithm such as AES.
*   **Use HTTPS:** Use HTTPS to encrypt communication between your game and your server.
*   **Protect API Keys:** Protect your API keys from being exposed by storing them securely on the server.
*   **Secure Local Storage:** If storing data locally, encrypt it to prevent unauthorized access.

### 4.5. Secure API Communication

*   **Use HTTPS:** Always use HTTPS for all API communication.
*   **Validate API Responses:** Validate API responses to ensure that they are valid and haven't been tampered with.
*   **Rate Limiting:** Implement rate limiting to prevent abuse of your API.
*   **API Key Rotation:** Rotate your API keys regularly to prevent them from being compromised.
*   **Proper Error Handling:** Implement proper error handling to prevent sensitive information from being leaked in error messages.

## 5. Testing Approaches

### 5.1. Unit Testing

*   **Test Individual Components:** Write unit tests to test individual components of your game, such as functions, classes, and nodes.
*   **Use a Testing Framework:** Use a testing framework such as Gut (Godot Unit Testing) to write and run your unit tests.
*   **Mock Dependencies:** Mock external dependencies to isolate the component being tested.
*   **Test Boundary Conditions:** Test boundary conditions and edge cases to ensure that your code handles them correctly.

### 5.2. Integration Testing

*   **Test Interactions Between Components:** Write integration tests to test the interactions between different components of your game.
*   **Use Realistic Scenarios:** Use realistic scenarios to test how the components work together in real-world situations.
*   **Verify Data Flow:** Verify that data flows correctly between the components being tested.

### 5.3. End-to-End Testing

*   **Test the Entire Game:** Write end-to-end tests to test the entire game from start to finish.
*   **Automate Testing:** Automate your end-to-end tests to ensure that they are run regularly.
*   **Use a Testing Tool:** Use a testing tool such as Selenium or Appium to automate your end-to-end tests.

### 5.4. Test Organization

*   **Create a `test/` Directory:** Create a `test/` directory in your project to store your tests.
*   **Mirror the Source Code Structure:** Mirror the source code structure in your test directory to make it easier to find the tests for a given component.
*   **Use Descriptive Names:** Use descriptive names for your tests to make it clear what they are testing.
*   **Keep Tests Short and Focused:** Keep your tests short and focused on testing a single aspect of the code.

### 5.5. Mocking and Stubbing

*   **Use Mocks to Isolate Components:** Use mocks to replace external dependencies with controlled test doubles.
*   **Use Stubs to Provide Predefined Responses:** Use stubs to provide predefined responses to external dependencies.
*   **Use a Mocking Framework:** Use a mocking framework such as Moq or NSubstitute to create mocks and stubs easily.

## 6. Common Pitfalls and Gotchas

### 6.1. Frequent Mistakes

*   **Forgetting to Connect Signals:**  Ensure that you connect signals to their handlers correctly. A common mistake is to connect the signal within the editor but forget to connect dynamically created signals in code.
*   **Incorrect Node Paths:** Double-check node paths when using `get_node()`. Incorrect paths will lead to `null` references and errors.
*   **Not Freeing Resources:** Remember to free resources when they are no longer needed to prevent memory leaks.
*   **Using Global Variables Excessively:** Avoid using global variables excessively, as this can make your code difficult to maintain and debug.
*   **Ignoring the Godot Documentation:** The Godot documentation is a valuable resource. Consult it frequently to learn about new features and best practices.

### 6.2. Edge Cases

*   **Floating-Point Precision:** Be aware of floating-point precision issues when comparing floating-point numbers. Use `is_equal_approx()` instead of `==`.
*   **Race Conditions:** Be aware of race conditions when using threads. Use locks and other synchronization primitives to prevent data corruption.
*   **Resource Loading Errors:** Handle resource loading errors gracefully.  Use `is_valid()` to check if a resource loaded successfully.
*   **Input Events:**  Understand the difference between different input event types (e.g., `InputEventKey`, `InputEventMouseButton`) and handle them accordingly.

### 6.3. Version-Specific Issues

*   **API Changes:** Be aware of API changes between Godot versions. Consult the changelog when upgrading to a new version.
*   **GDScript Compatibility:** Ensure that your GDScript code is compatible with the Godot version you are using.
*   **C# Version Compatibility:** Be aware of the C# version supported by your Godot version. Ensure your code is compatible. Older versions of Godot are tied to older C# versions, new versions are tied to .NET.

### 6.4. Compatibility Concerns

*   **Platform-Specific Code:** Use platform-specific code sparingly. Use conditional compilation or separate scripts for platform-specific functionality.
*   **Graphics Driver Compatibility:** Test your game on different graphics drivers to ensure that it works correctly.
*   **Hardware Compatibility:** Test your game on different hardware configurations to ensure that it runs smoothly.

### 6.5. Debugging Strategies

*   **Use the Godot Editor Debugger:** Use the Godot editor debugger to step through your code and inspect variables.
*   **Use `print()` Statements:** Use `print()` statements to log information to the console.
*   **Use the Remote Debugger:** Use the remote debugger to debug your game on a different device.
*   **Attach a C# Debugger:** Attach a C# debugger (e.g., Visual Studio Debugger) to your project to debug C# code.

## 7. Tooling and Environment

### 7.1. Recommended Development Tools

*   **Godot Editor:** The official Godot Engine editor.
*   **Visual Studio Code:** A popular code editor with excellent support for GDScript and C#.
*   **Visual Studio:** A powerful IDE for C# development, commonly used for Godot C# projects.
*   **Git:** A version control system for tracking changes to your code.
*   **GitHub/GitLab/Bitbucket:** Online repositories for storing and collaborating on your code.

### 7.2. Build Configuration

*   **Project Settings:** Configure your project settings carefully, including input maps, rendering settings, and export settings.
*   **Custom Build Steps:** Use custom build steps to automate tasks such as asset processing and code generation.
*   **Export Templates:** Create export templates for different platforms to optimize your game for each platform.

### 7.3. Linting and Formatting

*   **GDScript Linter:** Use a GDScript linter to check your code for style errors and potential problems.  Consider creating a custom linter rule set.
*   **C# Code Style:** Configure Visual Studio or VS Code to automatically format your C# code according to the C# coding conventions.
*   **EditorConfig:** Use an EditorConfig file to define coding style settings for your project.

### 7.4. Deployment

*   **Exporting:** Use Godot's export functionality to create builds for different platforms.
*   **Distribution Platforms:** Distribute your game through platforms such as Steam, Itch.io, Google Play Store, and Apple App Store.
*   **Consider platform-specific requirements:** Each platform has specific requirements for builds. For example, you will need developer accounts, store assets, and signed builds.
*   **Auto-updating:** Using auto-updating on desktop builds can improve the player experience for long running projects.

### 7.5. CI/CD Integration

*   **Automated Builds:** Set up a CI/CD pipeline to automatically build and test your game whenever you commit changes to your code repository.
*   **Automated Testing:** Integrate your unit tests, integration tests, and end-to-end tests into your CI/CD pipeline.
*   **Automated Deployment:** Automate the deployment of your game to different distribution platforms.
*   **Use cloud build services:** Services such as GitHub actions, GitLab CI, and cloud build are capable of fully automating the deployment process.

## 8. GDScript Style Guide Summary

This section summarizes key GDScript style guide recommendations.

*   **Formatting:**
    *   Use LF line endings, UTF-8 encoding, and tabs for indentation (editor default).
    *   Limit line length to 100 characters (preferably 80).
    *   Use one statement per line.
    *   Format multiline statements for readability using parentheses.
    *   Avoid unnecessary parentheses.
    *   Use plain English boolean operators (`and`, `or`, `not`).
    *   Use whitespace around operators and after commas.
    *   Use double quotes for strings unless single quotes avoid escapes.
    *   Don't omit leading or trailing zeros in floating-point numbers.
    *   Use lowercase for letters in hexadecimal numbers.
    *   Use underscores in literals for large numbers.
*   **Naming Conventions:**
    *   Files: `snake_case.gd`
    *   Classes: `PascalCase`
    *   Nodes: `PascalCase`
    *   Functions: `snake_case`
    *   Variables: `snake_case`
    *   Signals: `snake_case` (past tense)
    *   Constants: `CONSTANT_CASE`
    *   Enums: `PascalCase` (names), `CONSTANT_CASE` (members)
*   **Code Order (within a script):**
    1.  `@tool`
    2.  `class_name`
    3.  `extends`
    4.  Docstring (`## comment`)
    5.  Signals
    6.  Enums
    7.  Constants
    8.  `@export` variables
    9.  Public variables
    10. Private variables
    11. `@onready` variables
    12. `_init()`
    13. `_enter_tree()`
    14. `_ready()`
    15. Other virtual methods
    16. Public methods
    17. Private methods
    18. Subclasses
*   **Static Typing:**
    *   Use explicit type hints (`: Type`) when the type is ambiguous or for clarity.
    *   Use inferred types (`:=`) when the type is obvious from the assignment.

By following these best practices and coding standards, you can create more maintainable, efficient, and secure Godot Engine projects.