---
description: Comprehensive best practices and coding standards for Unreal Engine projects. Covers code organization, performance, security, testing, and common pitfalls to ensure maintainable, efficient, and robust game development.
globs: *.h,*.cpp,*.uasset,*.umap,*.ini
---
# Unreal Engine Best Practices and Coding Standards

This document outlines best practices and coding standards for Unreal Engine projects. Following these guidelines will help ensure maintainability, efficiency, and robustness.

## 1. Code Organization and Structure

### 1.1 Directory Structure

*   **`Source/`:** Contains all C++ source code.
    *   **`[GameName]/`:** Root directory for your game's code.
        *   **`Public/`:** Header files (.h) for classes and components.
        *   **`Private/`:** Implementation files (.cpp).
        *   **`[Feature]/`:** Subdirectories for specific game features (e.g., `Inventory`, `AI`, `UI`).
        *   **`Core/`:** Classes and functions shared across the project.
        *   **`UMG/`**:  Code related to Unreal Motion Graphics (UI)
*   **`Content/`:** Contains all assets (meshes, textures, materials, blueprints, etc.).
    *   **`[GameName]/`:** Root directory for your game's content.
        *   **`Characters/`:** Character assets.
        *   **`Environments/`:** Environment assets (meshes, textures, materials).
        *   **`UI/`:** User interface assets.
        *   **`Blueprints/`:** Blueprint classes and scripts.
        *   **`Materials/`:**  Material assets.
        *   **`Textures/`:**  Texture assets.
        *   **`Audio/`:** Audio assets.
        *   **`Animations/`:**  Animation assets.
*   **`Config/`:** Configuration files (e.g., `DefaultGame.ini`, `DefaultEngine.ini`).
*   **`Plugins/`:** Plugin code and assets.

### 1.2 File Naming Conventions

*   **C++ Classes:** Use descriptive names with prefixes indicating the class type.
    *   `A[ClassName]` for Actors (e.g., `ACharacter`, `APlayerController`).
    *   `U[ClassName]` for UObjects (e.g., `UInventoryComponent`, `UGameInstance`).
    *   `F[StructName]` for Structs (e.g., `FHitResult`, `FVector`).
    *   `E[EnumName]` for Enums (e.g., `EMovementMode`, `EInventoryItemType`).
    *   `I[InterfaceName]` for Interfaces (e.g., `IInteractable`, `IUsable`).
*   **Blueprints:**
    *   `BP_[AssetName]` (e.g., `BP_Character`, `BP_Door`).
*   **Assets:** Use descriptive names with prefixes indicating asset type.
    *   `SM_[AssetName]` for Static Meshes (e.g., `SM_Table`, `SM_Chair`).
    *   `T_[AssetName]` for Textures (e.g., `T_Ground_Albedo`, `T_Wall_Normal`).
    *   `M_[AssetName]` for Materials (e.g., `M_Rock`, `M_Water`).
    *   `MI_[AssetName]` for Material Instances (e.g., `MI_Rock_Dark`, `MI_Water_Shallow`).
    *   `S_[AssetName]` for Sounds (e.g., `S_Explosion`, `S_Footstep`).
    *   `Anim_[AssetName]` for Animations (e.g., `Anim_Run`, `Anim_Jump`).
    *   `Mat_[AssetName]` for Matinee Sequences (Legacy Animation).
*   **Levels:**
    *   `[LevelName]_Level` or `[LevelName]` (e.g., `MainMenu_Level`, `Gameplay`).

### 1.3 Module Organization

*   **Game Module:** The main module containing the game's core logic.
*   **Feature Modules:** Modules dedicated to specific game features (e.g., `InventoryModule`, `AIModule`).
*   **Plugin Modules:** Modules packaged as plugins, offering reusable functionality.

### 1.4 Component Architecture

*   **Favor Composition over Inheritance:** Use components to add functionality to actors.
*   **Create Reusable Components:** Design components to be generic and adaptable.
*   **Use Interfaces for Communication:** Define interfaces for components to interact with each other.
*   **Avoid God Components:** Break down complex functionality into smaller, more manageable components.
*   **Encapsulate Logic:** Keep component logic self-contained and avoid tight coupling.

### 1.5 Code Splitting Strategies

*   **Feature-Based Splitting:** Organize code by game features.
*   **Module-Based Splitting:** Create separate modules for distinct functionalities.
*   **Async Loading:** Load assets and levels asynchronously to avoid hitches.
*   **Level Streaming:** Divide large levels into smaller, streamed sub-levels.
*   **Object Pooling:** Reuse frequently created and destroyed objects to reduce garbage collection.

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns

*   **Singleton:** For classes with only one instance (e.g., `GameInstance`).
*   **Observer:** For event-driven communication between actors and components.
*   **Factory:** For creating objects without specifying their concrete classes.
*   **Command:** For encapsulating actions as objects, enabling undo/redo functionality.
*   **Strategy:** For defining a family of algorithms and making them interchangeable.
*   **Decorator:** For dynamically adding responsibilities to an object.
*   **Adapter:** For adapting the interface of a class to another interface clients expect.
*   **Object Pool:** Reuse objects to avoid frequent allocation and deallocation, especially useful for projectiles or particle effects.

### 2.2 Recommended Approaches for Common Tasks

*   **Actor Spawning:** Use `GetWorld()->SpawnActor<>()` to spawn actors dynamically.
*   **Input Handling:** Use Enhanced Input System for flexible and configurable input mappings.
*   **Timers:** Use `GetWorldTimerManager()` for timed events and actions.
*   **Collision Handling:** Use collision components and event delegates for collision detection.
*   **Networking:** Use Unreal Engine's built-in networking system for multiplayer games.
*   **Animation:** Leverage Animation Blueprints and State Machines for complex character animations.
*   **UI Design:**  Utilize UMG (Unreal Motion Graphics) for creating dynamic user interfaces.

### 2.3 Anti-patterns and Code Smells

*   **God Classes/Actors:** Avoid overly large classes with too many responsibilities.
*   **Spaghetti Code:** Avoid complex and unstructured code that is difficult to understand and maintain.
*   **Magic Numbers:** Avoid hardcoded values without clear explanations.
*   **Copy-Pasted Code:** Refactor duplicated code into reusable functions or classes.
*   **Tight Coupling:** Minimize dependencies between classes and components.
*   **Memory Leaks:** Ensure proper memory management to avoid memory leaks.
*   **Excessive Casting:** Minimize the use of `Cast<>` as it can impact performance.  Consider using interfaces or dynamic dispatch instead.
*   **Polling:** Avoid constantly checking for conditions; use events or delegates instead.
*   **Tick Abuse:** Avoid performing expensive operations in the `Tick()` function; use timers or asynchronous tasks.
*   **Not using the Actor Component System:** Neglecting to leverage components leads to monolithic, inflexible Actor classes.

### 2.4 State Management

*   **Game Instance:** For storing global game state that persists across levels.
*   **Game Mode:** For managing game rules and player interactions within a level.
*   **Player State:** For storing player-specific information (e.g., score, inventory).
*   **Actor State:** For storing the state of individual actors (e.g., health, position).
*   **Use Data Assets:**  For storing configurable game data (e.g., weapon stats, enemy parameters).
*   **State Tree:** For handling complex AI behavior.
*   **Gameplay Abilities:** For managing player abilities and interactions using the Gameplay Ability System (GAS).

### 2.5 Error Handling

*   **Use `ensure()` for Debug Assertions:** Use `ensure()` to check for conditions that should always be true during development.
*   **Use `check()` for Critical Assertions:** Use `check()` for conditions that must be true in all builds.
*   **Use `try-catch` for Exception Handling:** Use `try-catch` blocks to handle exceptions in critical code sections.
*   **Log Errors and Warnings:** Use `UE_LOG()` to log errors and warnings for debugging and monitoring.
*   **Handle Potential Null Pointers:** Always check for null pointers before accessing object members.
*   **Implement Recovery Mechanisms:** Provide mechanisms to recover from errors gracefully (e.g., retry logic, fallback behavior).
*   **Validate Data:**  Implement robust input and data validation to prevent unexpected errors.

## 3. Performance Considerations

### 3.1 Optimization Techniques

*   **Profiling:** Use Unreal Engine's profiling tools (e.g., Unreal Insights, Stat commands) to identify performance bottlenecks.
*   **LODs (Level of Detail):** Use LODs to reduce the complexity of meshes at a distance.
*   **Occlusion Culling:** Use occlusion culling to hide objects that are not visible.
*   **Distance Culling:** Use distance culling to remove objects that are too far away.
*   **HLODs (Hierarchical Level of Detail):** Use HLODs to combine multiple static meshes into a single mesh.
*   **Instanced Static Meshes:** Use instanced static meshes to render multiple copies of the same mesh efficiently.
*   **Material Optimization:** Optimize material complexity to reduce pixel cost.
*   **Texture Optimization:** Use appropriate texture resolutions and compression formats.
*   **Blueprint Nativization:** Convert Blueprint logic to C++ for improved performance.
*   **Asynchronous Loading:** Load assets and levels asynchronously to avoid hitches.
*   **Object Pooling:** Reuse frequently created and destroyed objects to reduce garbage collection.
*   **Avoid Dynamic Allocation:** Minimize dynamic memory allocation during runtime.
*   **Optimize Collision:** Simplify collision geometry and disable unnecessary collision checks.
*   **Use Niagara:** Use the Niagara particle system for efficient particle effects.
*   **Implement Adaptive Resolution:** Dynamically adjust the resolution based on performance.
*   **Use GPU Profiling tools:** Tools like RenderDoc can help you debug what is happening on the GPU and pin point bottlenecks.

### 3.2 Memory Management

*   **Use Smart Pointers:** Use `TSharedPtr` and `TWeakPtr` for automatic memory management.
*   **Avoid Circular Dependencies:** Prevent circular dependencies between smart pointers.
*   **Unload Unused Assets:** Unload assets that are no longer needed to free up memory.
*   **Garbage Collection:** Understand and optimize garbage collection behavior.
*   **Asset References:** Use asset references carefully to avoid unnecessary asset loading.
*   **Texture Streaming:** Use texture streaming to load only the necessary texture mipmaps.
*   **Minimize Asset Duplication:**  Avoid creating duplicate assets; reuse existing ones whenever possible.
*   **Monitor Memory Usage:**  Regularly monitor memory usage to identify potential leaks or excessive consumption.

### 3.3 Rendering Optimization

*   **Minimize Draw Calls:** Reduce the number of unique meshes and materials.
*   **Optimize Shaders:** Simplify shader complexity to reduce rendering time.
*   **Use Post-Processing Effects Sparingly:** Post-processing effects can be expensive; use them judiciously.
*   **Optimize Lighting:** Use baked lighting whenever possible to reduce runtime lighting calculations.
*   **Shadow Optimization:** Optimize shadow settings to reduce shadow rendering cost.
*   **Use Mobile Rendering Features:** Utilize mobile rendering features for improved performance on mobile devices.
*   **Consider Nanite Carefully:** For next-gen fidelity, use Nanite, but be aware of its overhead on lower-end platforms.
*   **Virtual Shadow Maps:** Use virtual shadow maps to improve shadow quality and performance in large open worlds.

### 3.4 Package Size Optimization

*   **Compress Assets:** Use asset compression to reduce package size.
*   **Remove Unused Assets:** Delete unused assets from the project.
*   **Texture Compression:** Use appropriate texture compression formats.
*   **Audio Compression:** Use appropriate audio compression formats.
*   **Blueprint Stripping:** Strip debug information from Blueprints in release builds.
*   **Cook Only Necessary Assets:** Ensure that only necessary assets are cooked for the target platform.
*   **Use Pak File Compression:** Employ compression when creating .pak files for deployment.

### 3.5 Lazy Loading

*   **Stream Levels:** Load and unload levels dynamically based on player location.
*   **Load Assets On-Demand:** Load assets only when they are needed.
*   **Use Async Load Asset:** Use the `Async Load Asset` node in Blueprints or `LoadObjectAsync` in C++ to load assets asynchronously.
*   **Lazy Loading Proxies:**  Create proxy objects that load the full asset only when accessed.
*   **Subobject loading:** Defer loading of certain UObject subobjects until needed.

## 4. Security Best Practices

### 4.1 Common Vulnerabilities

*   **Code Injection:** Prevent code injection by validating all input data.
*   **Denial of Service (DoS):** Protect against DoS attacks by limiting resource usage.
*   **Man-in-the-Middle (MitM):** Use encryption to protect data in transit.
*   **Data Tampering:** Prevent data tampering by using checksums and digital signatures.
*   **Save Game Manipulation:**  Protect save game data from modification to prevent cheating.

### 4.2 Input Validation

*   **Validate All Input Data:** Validate all input data from players and external sources.
*   **Sanitize Input Data:** Sanitize input data to remove potentially malicious characters or code.
*   **Limit Input Length:** Limit the length of input strings to prevent buffer overflows.
*   **Use Regular Expressions:** Use regular expressions to validate input patterns.
*   **Implement Whitelists:**  Use whitelists to define allowed characters and patterns.

### 4.3 Authentication and Authorization

*   **Use Secure Authentication Methods:** Use secure authentication methods such as OAuth 2.0 or JWT.
*   **Implement Role-Based Access Control (RBAC):** Use RBAC to control access to different features and resources.
*   **Use Strong Passwords:** Enforce the use of strong passwords.
*   **Implement Multi-Factor Authentication (MFA):** Use MFA for added security.
*   **Store Credentials Securely:**  Store user credentials securely using encryption and salting.
*   **Avoid Storing Secrets in Code:** Use configuration files or environment variables to store sensitive information.

### 4.4 Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data at rest and in transit.
*   **Use Secure Communication Protocols:** Use HTTPS for secure communication over the network.
*   **Protect Save Game Data:** Encrypt save game data to prevent cheating.
*   **Implement Data Backups:** Regularly back up data to prevent data loss.
*   **Comply with Data Privacy Regulations:**  Comply with data privacy regulations such as GDPR and CCPA.
*   **Avoid Exposing Debug Information in Release Builds:** Disable or remove debug information in release builds.

### 4.5 Secure API Communication

*   **Use API Keys:** Use API keys to authenticate API requests.
*   **Implement Rate Limiting:** Implement rate limiting to prevent abuse of API endpoints.
*   **Use Input Validation:** Utilize robust input validation on APIs
*   **Use Secure Communication Protocols:** Use HTTPS for secure communication over the network.
*   **Validate API Responses:** Validate API responses to ensure data integrity.
*   **Log API Requests and Responses:** Log API requests and responses for auditing and monitoring.

## 5. Testing Approaches

### 5.1 Unit Testing

*   **Test Individual Components:** Write unit tests for individual components and classes.
*   **Use a Testing Framework:** Use a testing framework such as Unreal Engine's built-in Automation System or third-party frameworks.
*   **Write Clear and Concise Tests:** Write tests that are easy to understand and maintain.
*   **Test Edge Cases:** Test edge cases and boundary conditions.
*   **Use Mock Objects:** Use mock objects to isolate components during testing.
*   **Test Driven Development:** Consider using Test Driven Development (TDD) principles.
*   **Focus on Core Logic:** Prioritize testing core logic and critical functionalities.

### 5.2 Integration Testing

*   **Test Interactions Between Components:** Write integration tests to test the interactions between different components and classes.
*   **Test Game Logic:** Test the overall game logic and flow.
*   **Test Data Flow:** Test the flow of data between different systems.
*   **Simulate Realistic Scenarios:**  Simulate realistic game scenarios during integration testing.

### 5.3 End-to-End Testing

*   **Test the Entire Game:** Test the entire game from start to finish.
*   **Use Automated Testing Tools:** Use automated testing tools to simulate player interactions.
*   **Test on Different Platforms:** Test on different platforms to ensure compatibility.
*   **Test with Real Players:**  Test with real players to get feedback on gameplay and usability.

### 5.4 Test Organization

*   **Organize Tests by Feature:** Organize tests by game features or modules.
*   **Create a Test Suite:** Create a test suite that can be run automatically.
*   **Use a Consistent Naming Convention:** Use a consistent naming convention for tests.
*   **Document Tests:** Document tests to explain their purpose and functionality.
*   **Keep Tests Up-to-Date:** Keep tests up-to-date with code changes.

### 5.5 Mocking and Stubbing

*   **Use Mock Objects:** Use mock objects to simulate dependencies during testing.
*   **Use Stub Functions:** Use stub functions to replace complex or external functions.
*   **Isolate Components:** Isolate components during testing to avoid unintended interactions.
*   **Use a Mocking Framework:** Use a mocking framework such as Google Mock or EasyMock.

## 6. Common Pitfalls and Gotchas

### 6.1 Frequent Mistakes

*   **Incorrect Asset Management:** Failing to properly manage and reference assets.
*   **Over-Reliance on Blueprints:** Overusing Blueprints for complex logic, leading to performance issues.
*   **Ignoring Performance Considerations:** Neglecting to optimize code and assets for performance.
*   **Poor Memory Management:** Creating memory leaks and excessive garbage collection.
*   **Inadequate Error Handling:** Failing to handle errors and exceptions gracefully.
*   **Lack of Testing:** Not writing adequate unit and integration tests.
*   **Incorrect use of transform:** Using world transform directly, instead of relative transform for attached components.
*   **Not using the Garbage Collector Properly:** Not understanding when and how the Unreal Engine garbage collector reclaims memory.

### 6.2 Edge Cases

*   **Platform-Specific Issues:** Encountering platform-specific bugs or limitations.
*   **Device-Specific Issues:** Encountering device-specific performance issues.
*   **Localization Issues:** Encountering issues with text localization and internationalization.
*   **Save Game Corruption:** Dealing with corrupted save game data.
*   **Network Latency:** Handling network latency and packet loss in multiplayer games.

### 6.3 Version-Specific Issues

*   **API Changes:** Dealing with API changes between Unreal Engine versions.
*   **Deprecated Features:** Using deprecated features that may be removed in future versions.
*   **Compatibility Issues:** Encountering compatibility issues between different Unreal Engine versions.
*   **Shader Model Differences:**  Accounting for differences in shader models across different UE versions.

### 6.4 Compatibility Concerns

*   **Plugin Compatibility:** Ensuring compatibility between different plugins.
*   **Hardware Compatibility:** Ensuring compatibility with different hardware configurations.
*   **Operating System Compatibility:** Ensuring compatibility with different operating systems.
*   **Third-Party Library Compatibility:** Ensuring compatibility with third-party libraries and SDKs.

### 6.5 Debugging Strategies

*   **Use the Unreal Engine Debugger:** Use the Unreal Engine debugger to step through code and inspect variables.
*   **Use Logging Statements:** Use logging statements to track the flow of execution and identify errors.
*   **Use Breakpoints:** Set breakpoints in the code to pause execution at specific points.
*   **Use the Visual Logger:** Use the Visual Logger to visualize game data and events.
*   **Use Unreal Insights:** Use Unreal Insights to profile performance and identify bottlenecks.
*   **Crash Reporting:**  Implement crash reporting to collect crash logs and diagnose issues.

## 7. Tooling and Environment

### 7.1 Recommended Development Tools

*   **Visual Studio:**  For C++ development on Windows.
*   **XCode:** For C++ development on macOS.
*   **Rider for Unreal Engine:** Cross-platform IDE for Unreal Engine development with C++ and Blueprints.
*   **Substance Painter:** For creating and texturing materials.
*   **Blender/Maya/3ds Max:** For creating 3D models and animations.
*   **Perforce/Git:** For version control.
*   **RenderDoc/PIX:** For GPU debugging and profiling.

### 7.2 Build Configuration

*   **Use Development Builds:** Use development builds for debugging and testing.
*   **Use Shipping Builds:** Use shipping builds for release.
*   **Configure Build Settings:** Configure build settings such as optimization level and debug information.
*   **Use Build Automation:** Use build automation tools to automate the build process.
*   **Set up Build Targets:** Properly configure build targets based on desired device platform (desktop, mobile, console, etc.)

### 7.3 Linting and Formatting

*   **Use a Code Linter:** Use a code linter to enforce code style and identify potential errors.
*   **Use a Code Formatter:** Use a code formatter to automatically format code.
*   **Follow the Unreal Engine Coding Standard:** Follow the Unreal Engine coding standard for consistency.
*   **Configure Editor Settings:** Configure editor settings to automatically format code on save.

### 7.4 Deployment

*   **Package the Game:** Package the game for the target platform.
*   **Test the Packaged Game:** Test the packaged game thoroughly before release.
*   **Use a Deployment Platform:** Use a deployment platform such as Steam or the Epic Games Store.
*   **Follow Platform-Specific Requirements:**  Follow platform-specific requirements for deployment.

### 7.5 CI/CD Integration

*   **Use a CI/CD System:** Use a CI/CD system such as Jenkins, Travis CI, or GitLab CI.
*   **Automate Builds and Tests:** Automate builds and tests as part of the CI/CD pipeline.
*   **Integrate with Version Control:** Integrate the CI/CD system with the version control system.
*   **Automate Deployment:** Automate deployment to the target platform as part of the CI/CD pipeline.
*   **Trigger Builds on Code Changes:**  Configure the CI/CD system to trigger builds automatically on code changes.