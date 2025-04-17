---
description: This rule provides best practices for Unity C# development, covering code style, organization, performance, and security to ensure maintainable and efficient projects.
globs: *.cs
---
# Unity C# Best Practices and Coding Standards

This document provides a comprehensive guide to best practices for Unity C# development, covering code style, organization, performance, and security, to ensure maintainable and efficient projects.

## I. Code Organization and Structure

### A. Directory Structure Best Practices

A well-organized directory structure is crucial for maintainability and collaboration. Consider the following structure:


Assets/
├── Animations/
│   ├── AnimationClips/
│   └── Animators/
├── Audio/
│   ├── Music/
│   └── SFX/
├── Editor/
│   └── EditorScripts/
├── Fonts/
├── Materials/
├── Models/
├── Plugins/
├── Prefabs/
├── Resources/
├── Scenes/
├── Scripts/
│   ├── Core/
│   ├── Gameplay/
│   ├── UI/
│   ├── Data/
│   ├── Editor/
│   └── Utilities/
├── Textures/
│   ├── UI/
│   └── Environment/


*   **Animations:** Contains all animation-related assets.
*   **Audio:** Contains music and sound effects.
*   **Editor:** Contains custom editor scripts.
*   **Fonts:** Contains font assets.
*   **Materials:** Contains material assets.
*   **Models:** Contains 3D models.
*   **Plugins:** Contains third-party plugins.
*   **Prefabs:** Contains prefab assets.
*   **Resources:** Contains assets loaded at runtime (use sparingly due to performance implications).
*   **Scenes:** Contains scene files.
*   **Scripts:** Contains all C# scripts, further organized by functionality.
    *   **Core:** Fundamental scripts and systems.
    *   **Gameplay:** Scripts related to gameplay mechanics.
    *   **UI:** User interface scripts.
    *   **Data:** Scripts related to data management (e.g., ScriptableObjects).
    *   **Editor:** Custom editor tools and scripts.
    *   **Utilities:** General-purpose utility scripts.
*   **Textures:** Contains texture assets.

### B. File Naming Conventions

Consistent file naming improves project readability and searchability.

*   **Scripts:** `PascalCase.cs` (e.g., `PlayerController.cs`)
*   **Prefabs:** `PascalCase.prefab` (e.g., `EnemyPrefab.prefab`)
*   **Scenes:** `PascalCase.unity` (e.g., `MainMenu.unity`)
*   **Materials:** `PascalCase.mat` (e.g., `WaterMaterial.mat`)
*   **Textures:** `PascalCase.png` or `PascalCase.jpg` (e.g., `GroundTexture.png`)
*   **Animations:** `PascalCase.anim` (e.g., `PlayerIdle.anim`)

Follow these conventions:

*   Use PascalCase for class names, methods, and properties.
*   Use camelCase for variables and parameters.
*   Use UPPER_SNAKE_CASE for constants.

### C. Module Organization Best Practices

For larger projects, consider organizing code into modules or namespaces.

*   **Namespaces:** Group related classes within a namespace.  This avoids naming collisions and improves code organization. Use namespaces that reflect the folder structure.

    csharp
    namespace MyGame.Gameplay
    {
        public class PlayerController : MonoBehaviour
        {
            // ...
        }
    }
    

*   **Assembly Definitions:** Use Assembly Definition files (`.asmdef`) to define modules.  This enables faster compilation times and better code isolation.  Place each module in its own folder with an assembly definition file.

### D. Component Architecture Recommendations

Unity uses a component-based architecture. Design your game objects with small, reusable components that handle specific responsibilities.

*   **Single Responsibility Principle:** Each component should have one specific responsibility.
*   **Composition over Inheritance:** Favor composition over inheritance to create complex behavior.

Example:

Instead of a monolithic `Player` script, use separate components like `PlayerMovement`, `PlayerHealth`, and `PlayerAttack`.

### E. Code Splitting Strategies

*   **Partial Classes:** Split large classes into multiple files using the `partial` keyword.

    csharp
    // PlayerController.cs
    public partial class PlayerController : MonoBehaviour
    {
        // Movement logic
    }

    // PlayerController.Combat.cs
    public partial class PlayerController : MonoBehaviour
    {
        // Combat logic
    }
    

*   **Extension Methods:** Add functionality to existing classes without modifying their source code.

    csharp
    public static class StringExtensions
    {
        public static string Capitalize(this string str)
        {
            return char.ToUpper(str[0]) + str.Substring(1);
        }
    }

    // Usage
    string name = "john";
    string capitalizedName = name.Capitalize(); // John
    

## II. Common Patterns and Anti-patterns

### A. Design Patterns

*   **Singleton:** Ensure a class has only one instance and provides a global point of access to it (use carefully, as overuse can lead to tight coupling).

    csharp
    public class GameManager : MonoBehaviour
    {
        private static GameManager _instance;
        public static GameManager Instance
        {
            get { return _instance; }
        }

        private void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(this.gameObject);
            } else {
                _instance = this;
                DontDestroyOnLoad(this.gameObject);
            }
        }
    }
    

*   **Object Pooling:** Reuse objects instead of creating and destroying them frequently to reduce garbage collection overhead.

    csharp
    public class ObjectPool : MonoBehaviour
    {
        public GameObject pooledObject;
        public int poolSize = 10;
        private List<GameObject> pool;

        void Start()
        {
            pool = new List<GameObject>();
            for (int i = 0; i < poolSize; i++)
            {
                GameObject obj = (GameObject)Instantiate(pooledObject);
                obj.SetActive(false);
                pool.Add(obj);
            }
        }

        public GameObject GetPooledObject()
        {
            for (int i = 0; i < pool.Count; i++)
            {
                if (!pool[i].activeInHierarchy)
                {
                    return pool[i];
                }
            }
            return null; // Or Instantiate more if needed
        }
    }
    
*   **Factory:** Create objects without specifying the exact class of object that will be created.
*   **Observer:** Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
*   **Command:** Encapsulate a request as an object, thereby allowing for parameterizing clients with queues, requests, and operations.

### B. Recommended Approaches for Common Tasks

*   **Input Handling:** Use the new Input System for more flexible and customizable input handling.
*   **UI Development:** Utilize Unity's UI system (Canvas, RectTransform, UI components) for creating user interfaces.
*   **Data Persistence:** Use `PlayerPrefs` for simple data storage, and consider using serialization for more complex data structures.  Alternatively, consider using a database like SQLite for more complex data storage needs.
*   **Networking:** Use Unity's `Netcode for GameObjects` for multiplayer games.  Also consider third-party networking solutions like Photon.
*   **Asynchronous Operations:** Employ `async/await` to avoid blocking the main thread when performing long-running operations (e.g., loading assets, networking).

### C. Anti-patterns and Code Smells

*   **God Classes:** Avoid creating classes that do too much.  Split functionality into smaller, more manageable classes.
*   **Spaghetti Code:** Avoid complex, unstructured code that is difficult to understand and maintain.  Use modular design and clear coding conventions.
*   **Magic Numbers:** Avoid hardcoded numerical values in your code. Use named constants instead.
*   **Overuse of `FindGameObjectWithTag` or `GetComponentInChildren`:** These methods can be slow. Cache references to frequently used objects and components.
*   **Using `Resources.Load` excessively:**  `Resources.Load`  can lead to performance issues. Use AssetBundles or Addressables for better asset management.
*   **Relying heavily on `Update()` for everything:**  Minimize the code in the  `Update()`  loop to avoid performance bottlenecks. Use events, coroutines, and other techniques to handle tasks outside of the main loop.

### D. State Management Best Practices

*   **State Machines:** Use state machines to manage complex object behavior.

    csharp
    public enum PlayerState
    {
        Idle,
        Walking,
        Jumping,
        Attacking
    }

    public class PlayerController : MonoBehaviour
    {
        public PlayerState currentState = PlayerState.Idle;

        void Update()
        {
            switch (currentState)
            {
                case PlayerState.Idle:
                    // Handle idle state logic
                    break;
                case PlayerState.Walking:
                    // Handle walking state logic
                    break;
                // ...
            }
        }
    }
    

*   **ScriptableObjects:** Use ScriptableObjects to store game data and configuration parameters.

    csharp
    [CreateAssetMenu(fileName = "WeaponData", menuName = "Game Data/Weapon Data", order = 1)]
    public class WeaponData : ScriptableObject
    {
        public string weaponName;
        public int damage;
        public float fireRate;
    }
    

### E. Error Handling Patterns

*   **Try-Catch Blocks:** Use try-catch blocks to handle exceptions gracefully.  Log exceptions and provide informative error messages.
*   **Assertions:** Use assertions to validate assumptions in your code.
*   **Null Checks:** Check for null references before accessing objects to prevent NullReferenceExceptions.
*   **Custom Exceptions:** Create custom exception classes to handle specific error conditions in your game.

## III. Performance Considerations

### A. Optimization Techniques

*   **Object Pooling:** Reuse objects to reduce garbage collection.
*   **Caching:** Cache frequently accessed data to avoid repeated calculations or lookups.
*   **String Concatenation:** Use `StringBuilder` for efficient string concatenation.
*   **Minimize Garbage Collection:** Avoid creating temporary objects in frequently executed code.
*   **Disable Unused Components:** Disable components that are not currently needed.
*   **Reduce Draw Calls:** Batch static objects, use texture atlases, and optimize materials to reduce draw calls.
*   **LOD (Level of Detail):** Use LOD groups to reduce the polygon count of objects at a distance.
*   **Occlusion Culling:** Occlude objects that are not visible to the camera.
*   **Use Profiler:** Regularly use the Unity Profiler to identify performance bottlenecks.

### B. Memory Management

*   **Asset Bundles:** Use Asset Bundles to load and unload assets dynamically, reducing the memory footprint of your game.
*   **Addressable Asset System:**  Use Addressables for an even more flexible asset management system.
*   **Unload Unused Assets:** Call `Resources.UnloadUnusedAssets()` to release unused assets from memory.
*   **Weak References:** Use weak references to avoid memory leaks when referencing objects that may be destroyed.

### C. Rendering Optimization

*   **Optimize Shaders:** Use simple shaders and avoid complex calculations in shaders.
*   **Texture Compression:** Compress textures to reduce memory usage and improve rendering performance.
*   **Mipmapping:** Use mipmaps to reduce aliasing and improve performance.
*   **Lightmapping:** Bake static lighting to reduce real-time lighting calculations.
*   **Shadows:** Optimize shadow settings and reduce the number of real-time shadows.
*   **Post-Processing:** Use post-processing effects sparingly, as they can be performance intensive.

### D. Bundle Size Optimization

*   **Texture Compression:** Compress textures to reduce their size.
*   **Audio Compression:** Compress audio files to reduce their size.
*   **Remove Unused Assets:** Delete unused assets from your project.
*   **Use Asset Bundles:**  Split your game into multiple Asset Bundles to allow users to download only the content they need.
*   **Stripping Level:**  Configure stripping level in project settings to remove unused code.

### E. Lazy Loading

*   **Load Assets Asynchronously:** Load assets in the background using `async/await` or coroutines.
*   **Load Scenes Additively:** Load scenes additively to avoid interrupting gameplay.
*   **Stream Assets:** Stream large assets from disk or the network instead of loading them into memory all at once.

## IV. Security Best Practices

### A. Common Vulnerabilities

*   **Code Injection:** Prevent code injection by validating user input and avoiding the use of `eval` or similar functions.
*   **Data Tampering:** Protect game data from tampering by using encryption and checksums.
*   **Man-in-the-Middle Attacks:** Use HTTPS for all network communication to prevent man-in-the-middle attacks.
*   **Denial of Service (DoS):** Protect your server from DoS attacks by implementing rate limiting and input validation.

### B. Input Validation

*   **Validate All User Input:** Validate all user input to prevent code injection, data tampering, and other attacks.
*   **Use Whitelisting:** Use whitelisting to allow only specific characters or values in user input.
*   **Limit Input Length:** Limit the length of user input to prevent buffer overflows.
*   **Sanitize Input:** Sanitize user input to remove potentially harmful characters or code.

### C. Authentication and Authorization

*   **Use Secure Authentication:** Use a secure authentication method such as OAuth 2.0 or JWT (JSON Web Tokens).
*   **Implement Authorization:** Implement authorization to control access to resources based on user roles and permissions.
*   **Store Passwords Securely:** Hash passwords using a strong hashing algorithm such as bcrypt or Argon2.
*   **Use Multi-Factor Authentication:** Use multi-factor authentication to add an extra layer of security.

### D. Data Protection

*   **Encrypt Sensitive Data:** Encrypt sensitive data such as passwords, API keys, and payment information.
*   **Use Secure Storage:** Store sensitive data in a secure storage location such as the keychain or a hardware security module (HSM).
*   **Obfuscate Code:** Obfuscate your code to make it more difficult for attackers to reverse engineer.

### E. Secure API Communication

*   **Use HTTPS:** Use HTTPS for all API communication.
*   **Validate API Responses:** Validate API responses to prevent data injection and other attacks.
*   **Use API Keys:** Use API keys to authenticate requests to your API.
*   **Implement Rate Limiting:** Implement rate limiting to prevent abuse of your API.

## V. Testing Approaches

### A. Unit Testing

*   **Isolate Components:** Write unit tests for individual components in isolation.
*   **Use a Testing Framework:** Use a unit testing framework such as NUnit or Unity Test Runner.
*   **Test Edge Cases:** Test edge cases and boundary conditions.
*   **Write Mock Objects:** Use mock objects to simulate dependencies.
*   **Follow AAA Pattern:** Arrange, Act, Assert.

### B. Integration Testing

*   **Test Interactions:** Test the interactions between multiple components.
*   **Use Test Scenes:** Create test scenes to isolate integration tests.
*   **Simulate Real-World Scenarios:** Simulate real-world scenarios to test the behavior of your game under different conditions.
*   **Use Data-Driven Tests:** Use data-driven tests to test multiple scenarios with different input data.

### C. End-to-End Testing

*   **Test the Entire Game Flow:** Test the entire game flow from start to finish.
*   **Use Automated Testing Tools:** Use automated testing tools such as Selenium or Appium.
*   **Test on Multiple Platforms:** Test your game on multiple platforms to ensure compatibility.
*   **Involve Testers:** Involve human testers to identify usability issues and other problems.

### D. Test Organization

*   **Create a Test Directory:** Create a separate directory for your tests.
*   **Mirror the Source Directory:** Mirror the structure of your source directory in your test directory.
*   **Name Test Classes Consistently:** Name your test classes consistently (e.g., `PlayerControllerTests`).
*   **Group Tests by Functionality:** Group your tests by functionality.

### E. Mocking and Stubbing

*   **Use Mocking Frameworks:** Use mocking frameworks such as Moq or NSubstitute.
*   **Create Interfaces:** Create interfaces for your dependencies to make them easier to mock.
*   **Avoid Hardcoded Dependencies:** Avoid hardcoded dependencies in your code.
*   **Use Dependency Injection:** Use dependency injection to inject mock objects into your code.

## VI. Common Pitfalls and Gotchas

### A. Frequent Mistakes

*   **Ignoring Performance:** Neglecting performance optimization from the beginning of the project.
*   **Overcomplicating Code:** Writing complex code when simpler solutions exist.
*   **Not Using Version Control:** Failing to use version control (e.g., Git) to manage code changes.
*   **Poor Asset Management:** Poorly organizing and managing assets, leading to project bloat.
*   **Neglecting Testing:** Not writing unit tests, integration tests, or end-to-end tests.
*   **Misunderstanding Coroutines:**  Improper use or overuse of coroutines leading to unexpected behavior or memory leaks.
*   **Incorrect Use of `Time.deltaTime`:** Using `Time.deltaTime` incorrectly, leading to frame-rate dependent behavior.

### B. Edge Cases

*   **Floating Point Precision:** Be aware of floating-point precision issues when comparing floating-point numbers.
*   **Garbage Collection Spikes:** Be aware of garbage collection spikes and try to minimize garbage generation.
*   **Platform Differences:** Test your game on different platforms to ensure compatibility.
*   **Screen Size and Resolution:**  Handle different screen sizes and resolutions gracefully.

### C. Version-Specific Issues

*   **API Changes:** Be aware of API changes between different versions of Unity.
*   **Bug Fixes:** Be aware of bug fixes in different versions of Unity.
*   **Feature Deprecations:** Be aware of feature deprecations in different versions of Unity.

### D. Compatibility Concerns

*   **.NET Framework:** Be aware of the .NET Framework version used by your project.
*   **Third-Party Plugins:** Ensure that third-party plugins are compatible with your version of Unity.
*   **Platform SDKs:** Ensure that the platform SDKs you are using are compatible with your version of Unity.

### E. Debugging Strategies

*   **Use the Unity Debugger:** Use the Unity debugger to step through your code and inspect variables.
*   **Use Debug.Log:** Use `Debug.Log` to print messages to the console.
*   **Use Assertions:** Use assertions to validate assumptions in your code.
*   **Use the Unity Profiler:** Use the Unity Profiler to identify performance bottlenecks.
*   **Remote Debugging:**  Use remote debugging to debug your game on a device.

## VII. Tooling and Environment

### A. Recommended Development Tools

*   **Visual Studio or Visual Studio Code:** Use a powerful IDE for C# development.
*   **Unity Asset Store:** Explore the Unity Asset Store for useful tools and assets.
*   **Version Control System (Git):** Use a version control system to manage code changes.
*   **Project Management Tool (Jira, Trello):** Use a project management tool to track tasks and bugs.
*   **Code Editor Extensions:** Use code editor extensions for linting, formatting, and code completion.

### B. Build Configuration

*   **Use Development Builds:** Use development builds for testing and debugging.
*   **Use Release Builds:** Use release builds for production deployments.
*   **Configure Build Settings:** Configure build settings such as scripting backend, target architecture, and optimization level.
*   **Use Scripting Define Symbols:** Use scripting define symbols to enable or disable code based on the build configuration.

### C. Linting and Formatting

*   **Use StyleCop Analyzers:** Use StyleCop Analyzers to enforce coding style rules.
*   **Use EditorConfig:** Use EditorConfig to define coding style settings for your project.
*   **Use a Code Formatter:** Use a code formatter to automatically format your code.

### D. Deployment Best Practices

*   **Test on Target Platforms:** Test your game on the target platforms before deployment.
*   **Submit to App Stores:** Submit your game to the appropriate app stores.
*   **Monitor Performance:** Monitor the performance of your game after deployment.
*   **Gather User Feedback:** Gather user feedback to improve your game.

### E. CI/CD Integration

*   **Use a CI/CD Platform:** Use a CI/CD platform such as Jenkins, Travis CI, or GitHub Actions.
*   **Automate Builds and Tests:** Automate builds and tests to ensure code quality.
*   **Automate Deployments:** Automate deployments to streamline the release process.
*   **Integrate with Version Control:** Integrate your CI/CD pipeline with your version control system.

By following these best practices, you can create maintainable, efficient, and secure Unity C# projects.