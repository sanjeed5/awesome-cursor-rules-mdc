---
description: This guide provides opinionated, actionable best practices for Unity C# development, focusing on code organization, performance, and common pitfalls to ensure a clean, maintainable, and performant codebase.
globs: **/*
---
# unity Best Practices

This document outlines our definitive guidelines for Unity C# development. Adhere to these rules to maintain a high-quality, performant, and collaborative codebase.

## 1. Code Organization & Project Structure

### 1.1 Project Folder Hierarchy
Organize assets by type within a `_Project` root, separating custom assets from third-party packages. Avoid root-level clutter.

❌ BAD
```
Assets/
  MyScript.cs
  PlayerModel.fbx
  Level1.unity
  ThirdPartyAsset/
    ...
```

✅ GOOD
```
Assets/
  _Project/
    Audio/
    Materials/
    Models/
    Prefabs/
    Scenes/
    Scripts/
      Core/
      Gameplay/
      UI/
    Textures/
  Plugins/          // Reserved for external plugins
  Standard Assets/  // Reserved for Unity's standard assets
  ThirdPartyAsset/  // Leave imported packages intact
```

### 1.2 Scene Hierarchy Structure
Prefab everything. Group GameObjects logically within scenes for clarity and easier navigation.

❌ BAD
```
- Player
- Enemy1
- UI_HealthBar
- MainCamera
- Light
- Environment_Tree
```

✅ GOOD
```
- Managers
  - GameManager
  - AudioManager
- Cameras
  - MainCamera
- Lights
  - Directional Light
- UI
  - Canvas
    - HUD
    - PauseMenu
- World
  - Terrain
  - Environment
    - Trees
    - Rocks
- Gameplay
  - Actors
    - Player
    - Enemy (Prefab instances)
  - Items
    - Collectible (Prefab instances)
- _DynamicObjects // Parent for all runtime instantiated objects
```

### 1.3 Naming Conventions
Use PascalCase for public members and filenames, camelCase for private fields and local variables. Prefix private fields with `_`.

❌ BAD
```csharp
public class player_controller : MonoBehaviour {
    public float movementSpeed;
    private GameObject target;
    void Start() { }
}
```

✅ GOOD
```csharp
public class PlayerController : MonoBehaviour
{
    [SerializeField] private float _movementSpeed = 5f; // Serialized private field
    private GameObject _target; // Private field
    public float MovementSpeed => _movementSpeed; // Public property

    void Start()
    {
        // Local variable
        GameObject tempObject = new GameObject("Temp");
    }
}
```

## 2. Common Patterns & Anti-patterns

### 2.1 Explicit Access Modifiers
Always declare access modifiers (`public`, `private`, `protected`). Do not rely on default `private`.

❌ BAD
```csharp
class MyClass {
    int value; // Defaults to private, but implicit
    void DoSomething() { }
}
```

✅ GOOD
```csharp
public class MyClass
{
    private int _value;
    public void DoSomething() { }
}
```

### 2.2 XML Documentation
Document all public classes, methods, and fields using XML `///` comments.

❌ BAD
```csharp
public class PlayerHealth : MonoBehaviour {
    public int currentHealth;
    public void TakeDamage(int amount) { /* ... */ }
}
```

✅ GOOD
```csharp
/// <summary>
/// Manages the player's health and handles damage/healing.
/// </summary>
public class PlayerHealth : MonoBehaviour
{
    /// <summary>The current health of the player.</summary>
    public int CurrentHealth { get; private set; } = 100;

    /// <summary>
    /// Applies damage to the player, reducing their health.
    /// </summary>
    /// <param name="amount">The amount of damage to apply.</param>
    public void TakeDamage(int amount)
    {
        CurrentHealth -= amount;
        if (CurrentHealth <= 0)
        {
            Debug.Log("Player defeated!");
        }
    }
}
```

### 2.3 Singleton Access
Cache references to singletons or use a dedicated manager. Avoid `FindObjectOfType` in `Update` or frequently called methods.

❌ BAD
```csharp
public class GameManager : MonoBehaviour {
    void Update() {
        // Repeatedly calling FindObjectOfType is a performance drain
        PlayerController player = FindObjectOfType<PlayerController>();
        if (player != null) {
            // ...
        }
    }
}
```

✅ GOOD
```csharp
// PlayerController.cs
public class PlayerController : MonoBehaviour
{
    public static PlayerController Instance { get; private set; }

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
        }
        else
        {
            Instance = this;
        }
    }
}

// GameManager.cs
public class GameManager : MonoBehaviour
{
    private PlayerController _player;

    void Start()
    {
        _player = PlayerController.Instance; // Cache reference once
    }

    void Update()
    {
        if (_player != null)
        {
            // Use the cached reference
        }
    }
}
```

## 3. Performance Considerations

### 3.1 String vs. ID Lookups
Always use ID-based lookups for Animator, Material, and Shader properties to avoid runtime string hashing.

❌ BAD
```csharp
animator.SetBool("IsWalking", true);
material.SetFloat("_Glossiness", 0.5f);
```

✅ GOOD
```csharp
private static readonly int _isWalkingHash = Animator.StringToHash("IsWalking");
private static readonly int _glossinessID = Shader.PropertyToID("_Glossiness");

void SetProperties(Animator animator, Material material)
{
    animator.SetBool(_isWalkingHash, true);
    material.SetFloat(_glossinessID, 0.5f);
}
```

### 3.2 Garbage Collection (GC) Avoidance
Minimize per-frame allocations. Use object pooling, reuse collections, and avoid unnecessary boxing or string concatenations in hot paths.

#### 3.2.1 Object Pooling
Reuse frequently instantiated objects (e.g., projectiles, particles) instead of destroying and recreating them.

❌ BAD
```csharp
void FireProjectile() {
    Instantiate(projectilePrefab, transform.position, Quaternion.identity);
}
```

✅ GOOD
```csharp
using UnityEngine.Pool; // Requires Unity 2021+

public class ProjectileSpawner : MonoBehaviour
{
    public GameObject projectilePrefab;
    private ObjectPool<GameObject> _projectilePool;

    void Awake()
    {
        _projectilePool = new ObjectPool<GameObject>(
            () => Instantiate(projectilePrefab),
            (obj) => obj.SetActive(true),
            (obj) => obj.SetActive(false),
            (obj) => Destroy(obj),
            collectionCheck: false,
            defaultCapacity: 20,
            maxSize: 100
        );
    }

    public void FireProjectile()
    {
        GameObject projectile = _projectilePool.Get();
        projectile.transform.position = transform.position;
        // ... set up projectile, then return to pool when done
    }
}
```

#### 3.2.2 Collection Reuse
Declare collections once and clear them for reuse instead of allocating new ones in loops.

❌ BAD
```csharp
void Update() {
    List<Enemy> enemiesInRange = new List<Enemy>(); // Allocates every frame
    // ... populate and use list
}
```

✅ GOOD
```csharp
private readonly List<Enemy> _enemiesInRange = new List<Enemy>(); // Allocated once

void Update() {
    _enemiesInRange.Clear(); // Reuse existing list
    // ... populate and use list
}
```

#### 3.2.3 Non-Allocating Physics APIs
Use `NonAlloc` versions of physics queries to prevent GC spikes.

❌ BAD
```csharp
RaycastHit[] hits = Physics.RaycastAll(transform.position, transform.forward, 10f);
```

✅ GOOD
```csharp
private RaycastHit[] _raycastHits = new RaycastHit[10]; // Pre-allocate array

void CheckForHits() {
    int numHits = Physics.RaycastNonAlloc(transform.position, transform.forward, _raycastHits, 10f);
    for (int i = 0; i < numHits; i++) {
        // Process _raycastHits[i]
    }
}
```

### 3.3 `UnityEngine.Object` Null Comparisons
Comparisons against `UnityEngine.Object` subclasses are more expensive than plain C# objects. Cache references and avoid in tight loops.

❌ BAD
```csharp
void Update() {
    if (myGameObject != null) { // Expensive null check every frame
        // ...
    }
}
```

✅ GOOD
```csharp
private GameObject _myGameObject; // Reference cached in Start/Awake

void Start() {
    _myGameObject = GameObject.Find("MyObject");
}

void Update() {
    if (_myGameObject) { // Unity's implicit bool conversion is optimized
        // ...
    }
}
```

### 3.4 Debug Logging
Wrap debug logs in `[Conditional]` attributes to strip them from non-development builds, preventing performance overhead and log spam in production.

❌ BAD
```csharp
void Update() {
    Debug.Log("Player position: " + transform.position); // Always included
}
```

✅ GOOD
```csharp
using System.Diagnostics; // For [Conditional]
using UnityEngine;

public static class AppLogger
{
    [Conditional("ENABLE_LOGS")] // Define ENABLE_LOGS in Player Settings for debug builds
    public static void Log(string message)
    {
        Debug.Log($"[{Time.frameCount}] {message}");
    }

    [Conditional("ENABLE_LOGS")]
    public static void LogWarning(string message)
    {
        Debug.LogWarning($"[{Time.frameCount}] {message}");
    }
}

public class PlayerMovement : MonoBehaviour
{
    void Update()
    {
        AppLogger.Log("Player position: " + transform.position); // Stripped in release builds
    }
}
```

## 4. Common Pitfalls & Gotchas

### 4.1 Spaces in File/Folder Names
Avoid spaces in asset names. Unity's command-line tools and some platforms have issues with paths containing spaces. Use PascalCase.

❌ BAD
```
Assets/My Game Scene.unity
Assets/Player Controller.cs
```

✅ GOOD
```
Assets/MyGameScene.unity
Assets/PlayerController.cs
```

### 4.2 Modifying Third-Party Assets
Never modify imported third-party asset files directly. Copy them to your `_Project` folder and modify the copy. This prevents update conflicts.

❌ BAD
```
// Directly modifying a script inside Assets/ThirdPartyAsset/Scripts/EnemyAI.cs
namespace ThirdParty {
    public class EnemyAI : MonoBehaviour {
        // Your custom changes here
    }
}
```

✅ GOOD
```
// Create Assets/_Project/Scripts/CustomEnemyAI.cs
namespace MyGame {
    public class CustomEnemyAI : ThirdParty.EnemyAI { // Inherit if possible
        // Your custom changes/overrides here
    }
}
// Or, if inheritance isn't suitable, copy the file and modify the copy in _Project.
```