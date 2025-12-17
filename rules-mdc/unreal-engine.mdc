---
description: Definitive guidelines for writing consistent, performant, and maintainable C++ and Blueprint code in Unreal Engine 5.x projects.
globs: **/*
---
# unreal-engine Best Practices

This guide outlines the essential coding standards and architectural patterns for developing robust and efficient Unreal Engine 5.x projects. Adhere to these rules to ensure high-quality, collaborative, and performant game development.

## 1. Code Organization & Structure

### 1.1 Naming Conventions (C++)

Always follow Epic's and Splash Damage's C++ naming conventions for clarity and consistency.

*   **Classes/Structs/Enums:** Use prefixes (`A`, `U`, `F`, `E`, `T`) followed by `PascalCase`.
*   **Member Variables:** Use `CamelCase`.
*   **Local Variables/Parameters:** Use `lower_snake_case`.
*   **Functions/Methods:** Use `PascalCase`.
*   **Boolean Variables:** Prefix with `b` (Epic) or `Is` (Verse-inspired).
*   **Event Handlers:** Prefix with `On`.
*   **Resource Creation/Destruction:** Use `Create`/`Destroy`.

❌ **BAD**
```cpp
// Class
class my_actor_class : public AActor {};
// Variable
float playerHealth;
// Function
void get_player_name();
// Enum
enum E_WEAPON_TYPE { ... };
// Boolean
bool is_dead_player;
```

✅ **GOOD**
```cpp
// Class
class AMyActorClass : public AActor {};
// Variable
float PlayerHealth; // Member variable
float new_health_value; // Local variable/parameter
// Function
void GetPlayerName() const;
// Enum
enum EWeaponType : uint8 { EWeaponType_Sword, EWeaponType_Bow };
// Boolean
bool bIsDead; // or IsDead()
// Event Handler
void OnPlayerDied();
// Resource Management
UMyWidget* CreatePlayerHUD();
void DestroyPlayerHUD(UMyWidget* WidgetToDestroy);
```

### 1.2 Naming Conventions (Assets & Blueprints)

Prefix all assets and Blueprints with their type to aid discoverability and organization.

*   **Blueprint Classes:** `BP_` (e.g., `BP_PlayerCharacter`)
*   **Materials:** `M_` (e.g., `M_Rock`)
*   **Material Instances:** `MI_` (e.g., `MI_Rock_Mossy`)
*   **Static Meshes:** `SM_` (e.g., `SM_Door`)
*   **Skeletal Meshes:** `SK_` (e.g., `SK_Character`)
*   **Textures:** `T_` (e.g., `T_Grass_D` for Diffuse, `T_Grass_N` for Normal)
*   **Particle Systems:** `P_` (e.g., `P_Explosion`)
*   **User Widgets:** `W_` (e.g., `W_MainMenu`)
*   **Gameplay Abilities:** `GA_` (e.g., `GA_Dash`)
*   **Gameplay Effects:** `GE_` (e.g., `GE_DamageOverTime`)
*   **Input Actions:** `IA_` (e.g., `IA_Jump`)

❌ **BAD**
```
Content/PlayerCharacter.uasset
Content/RockMaterial.uasset
Content/Explosion.uasset
```

✅ **GOOD**
```
Content/MyGame/Characters/Player/BP_PlayerCharacter.uasset
Content/MyGame/Environment/Materials/M_Rock.uasset
Content/MyGame/FX/P_Explosion.uasset
```

### 1.3 Folder Structure

Organize content by feature or domain, not by asset type. This improves project navigation and simplifies migration. Use `Developer-folder` for temporary, personal assets.

❌ **BAD**
```
Content/Materials/M_Rock.uasset
Content/Blueprints/BP_PlayerCharacter.uasset
Content/FX/P_Explosion.uasset
```

✅ **GOOD**
```
Content/MyGame/Environment/Props/Rock/M_Rock.uasset
Content/MyGame/Characters/Player/BP_PlayerCharacter.uasset
Content/MyGame/Weapons/Rifle/SK_Rifle.uasset
Content/MyGame/Developer/MyName/BP_TestActor.uasset
```

### 1.4 C++ vs Blueprint Responsibility

Delegate responsibilities based on performance and iteration needs.

*   **C++:** Implement performance-critical logic, complex algorithms, core systems (networking, AI pathfinding), and engine integrations.
*   **Blueprint:** Use for gameplay logic, data-driven iteration, UI, and visual scripting where performance is not paramount. Expose minimal, well-defined C++ functions and properties to Blueprint.

❌ **BAD**
```cpp
// C++: Hardcoding specific gameplay rules that designers need to tweak frequently
void AGameModeBase::BeginPlay()
{
    Super::BeginPlay();
    // Complex hardcoded spawn logic
    SpawnEnemyAtLocation(FVector(0,0,0));
}
```
```blueprint
// Blueprint: Implementing a complex, frequently-called pathfinding algorithm
// (This will be slow)
```

✅ **GOOD**
```cpp
// C++: Core system, exposed to Blueprint
UCLASS()
class MYGAME_API AEnemySpawner : public AActor
{
    GENERATED_BODY()
public:
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spawning")
    TSubclassOf<AEnemyCharacter> EnemyClassToSpawn;

    UFUNCTION(BlueprintCallable, Category = "Spawning")
    void SpawnEnemy(FVector Location);
};
```
```blueprint
// Blueprint: Data-driven iteration, using C++ core
// In BP_GameMode
// Event BeginPlay
// -> Create Enemy Spawner (C++ class)
// -> Set EnemyClassToSpawn (Blueprint-defined enemy BP)
// -> Call SpawnEnemy (C++ function)
```

## 2. Common Patterns & Anti-patterns

### 2.1 Smart Pointers

Always use Unreal's smart pointers (`TSharedPtr`, `TWeakPtr`, `TUniquePtr`) for heap-allocated objects that do not derive from `UObject`. This prevents memory leaks and manages object lifetimes.

❌ **BAD**
```cpp
FMyClass* MyRawPtr = new FMyClass();
// ... later, forgetting to delete
delete MyRawPtr;
```

✅ **GOOD**
```cpp
TSharedPtr<FMyClass> MySmartPtr = MakeShared<FMyClass>();
// MySmartPtr will be automatically managed
```

### 2.2 Const-Correctness

Mark functions and parameters `const` when they do not modify the object's state. This improves code safety and allows for better compiler optimizations.

❌ **BAD**
```cpp
// Function might accidentally modify state
void GetPlayerName(FString& OutName);
```

✅ **GOOD**
```cpp
// Clearly indicates no state modification
FString GetPlayerName() const;
// Parameter is read-only, no accidental modification
void ProcessData(const FMyData& Data);
```

### 2.3 `UFUNCTION` Specifiers

Use `BlueprintCallable` for functions that modify state or perform actions. Use `BlueprintPure` for functions that are getters and do not modify state. `BlueprintPure` functions appear as pure nodes in Blueprints.

❌ **BAD**
```cpp
UFUNCTION(BlueprintCallable)
FString GetPlayerName(); // Should be pure if it doesn't modify state

UFUNCTION(BlueprintPure)
void SetPlayerHealth(float NewHealth); // Should be callable as it modifies state
```

✅ **GOOD**
```cpp
UFUNCTION(BlueprintPure, Category = "Player")
FString GetPlayerName() const;

UFUNCTION(BlueprintCallable, Category = "Player")
void SetPlayerHealth(float NewHealth);
```

## 3. Performance Considerations

### 3.1 Avoid Heavy Logic in `Tick`

The `Tick` function executes every frame. Move expensive operations to timers, async tasks, or event-driven logic to prevent performance bottlenecks.

❌ **BAD**
```cpp
void AMyActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    // Expensive raycast every single frame
    FHitResult Hit;
    GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility);
    // Complex AI calculations
    PerformComplexAIBehavior();
}
```

✅ **GOOD**
```cpp
void AMyActor::BeginPlay()
{
    Super::BeginPlay();
    // Perform expensive operations periodically
    GetWorldTimerManager().SetTimer(MyTimerHandle, this, &AMyActor::PerformPeriodicCheck, 1.0f, true);
}

void AMyActor::PerformPeriodicCheck()
{
    FHitResult Hit;
    GetWorld()->LineTraceSingleByChannel(Hit, Start, End, ECC_Visibility);
}

// AI behavior triggered by events or specific conditions, not every frame
void AMyActor::OnTargetDetected()
{
    PerformComplexAIBehavior();
}
```

### 3.2 XR Optimization

For VR/AR/MR projects, follow Epic's XR Best Practices to ensure optimal performance and user comfort.

*   **Project Settings:** Start with the VR Template or `Scalable` quality, `Raytracing: Disabled`, `Starter Content: Disabled`. Enable `Forward Shading`, `MSAA`, `Instanced Stereo`, and `Mobile Multi-View` (for mobile VR).
*   **`.ini` Settings:** Set `vr.PixelDensity=1`, `r.SeparateTranslucency=0`, `r.HZBOcclusion=0` in `DefaultEngine.ini` under `[SystemSettings]`.
*   **Frame Rate:** Disable `Smooth Frame Rate` and `Use Fixed Frame Rate` in Project Settings > Engine > General Settings > Framerate.
*   **Nanite Foliage:** Utilize Nanite Foliage for dense, performant environments, especially for moving plants.

❌ **BAD**
```ini
// DefaultEngine.ini
[SystemSettings]
r.SeparateTranslucency=1 // Expensive for mobile VR
r.HZBOcclusion=1 // Can be disabled for perf
```

✅ **GOOD**
```ini
// DefaultEngine.ini
[SystemSettings]
vr.PixelDensity=1
r.SeparateTranslucency=0
r.HZBOcclusion=0
r.MotionBlurQuality=0
r.PostProcessAAQuality=3
r.BloomQuality=1
r.EyeAdaptationQuality=0
r.AmbientOcclusionLevels=0
r.SSR.Quality=1
r.DepthOfFieldQuality=0
r.SceneColorFormat=2
r.TranslucencyVolumeBlur=0
r.TranslucencyLightingVolumeDim=4
r.MaxAnisotropy=8
r.LensFlareQuality=0
r.SceneColorFringeQuality=0
r.FastBlurThreshold=0
r.SSR.MaxRoughness=0.1
r.rhicmdbypass=0
sg.EffectsQuality=2
sg.PostProcessQuality=0
```

## 4. Common Pitfalls & Gotchas

### 4.1 Blueprint Overuse for Performance

Avoid implementing complex, frequently-called, or performance-critical logic exclusively in Blueprints. Profile your game to identify bottlenecks.

❌ **BAD**
```blueprint
// In a Tick event or a loop that runs many times
// -> Complex mathematical calculations or array operations
// -> Heavy string manipulations
```

✅ **GOOD**
```cpp
// C++: Implement the core complex logic
UFUNCTION(BlueprintCallable, Category = "Calculations")
float CalculateComplexValue(float Input);
```
```blueprint
// Blueprint: Call the optimized C++ function
// -> Call CalculateComplexValue (C++ function)
```

### 4.2 Unnecessary `UObject` Inheritance

Only inherit from `UObject` (or its derivatives like `AActor`, `UDataAsset`, `UWidget`) when you need Unreal's reflection system, garbage collection, editor integration, or Blueprint exposure. For simple helper classes or data structures, use plain C++ classes or structs.

❌ **BAD**
```cpp
// UCLASS for a simple utility that doesn't need reflection or GC
UCLASS()
class UMySimpleHelper
{
    GENERATED_BODY()
public:
    void DoSomething();
};
```

✅ **GOOD**
```cpp
// Plain C++ class for a simple utility
class FMySimpleHelper
{
public:
    void DoSomething();
};
```

## 5. Testing Approaches

### 5.1 Automated Tests

Implement automated tests for all critical C++ logic using Unreal's Automation Test Framework. This ensures code correctness and prevents regressions.

*   **Unit Tests:** Verify individual functions and classes.
*   **Functional Tests:** Validate gameplay systems and interactions.

❌ **BAD**
```cpp
// No automated tests, relying solely on manual playtesting
```

✅ **GOOD**
```cpp
// MyGame.Build.cs (add modules for testing)
PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore" });
PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd", "FunctionalTesting", "AutomationController" });

// MyGameTest.cpp (example unit test)
#include "CoreMinimal.h"
#include "Misc/AutomationTest.h"
#include "MyGame/Public/MyMathLibrary.h" // Assuming this is your class

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FMyMathLibraryAddTest, "MyGame.Math.Add", EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::ProductFilter)

bool FMyMathLibraryAddTest::RunTest(const FString& Parameters)
{
    // Arrange
    int A = 5;
    int B = 10;

    // Act
    int Result = UMyMathLibrary::Add(A, B);

    // Assert
    TestEqual(TEXT("5 + 10 should be 15"), Result, 15);
    return true;
}
```

### 5.2 Blueprint Validation

Add robust validation within Blueprints, especially for data assets, critical paths, and user inputs. Use `IsValid` checks and `ensure`/`check` in C++ code exposed to Blueprint.

❌ **BAD**
```blueprint
// Assuming 'MyActorReference' is always valid
// -> MyActorReference.DoSomething()
```

✅ **GOOD**
```blueprint
// Always check validity before dereferencing
// Input: MyActorReference (Actor Object Reference)
// -> IsValid (MyActorReference)
// --- IF TRUE ---
// -> MyActorReference.DoSomething()
// --- IF FALSE ---
// -> Print String "Error: MyActorReference is null!"
```