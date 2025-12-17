---
description: This guide provides opinionated, actionable best practices for Godot 4.x development, focusing on code organization, common patterns, performance, and avoiding pitfalls.
globs: **/*
---
# godot Best Practices

This document outlines the definitive best practices for developing with Godot Engine 4.x. Adhere to these guidelines to ensure maintainable, performant, and collaborative projects.

## 1. Code Organization and Structure

### 1.1. Scene-First Composition
Always prefer composing scenes over deep inheritance hierarchies. Scenes are Godot's fundamental building blocks, promoting reusability and clear ownership.

❌ **BAD:** Deep inheritance
```gdscript
# res://entities/base_enemy.gd
class_name BaseEnemy extends CharacterBody2D
# ... common enemy logic ...

# res://entities/goblin.gd
class_name Goblin extends BaseEnemy
# ... goblin specific logic ...

# res://entities/orc.gd
class_name Orc extends BaseEnemy
# ... orc specific logic ...
```

✅ **GOOD:** Composition with specialized scenes
```gdscript
# res://components/health_component.gd
class_name HealthComponent extends Node
@export var max_health: int = 100
# ... health logic, signals ...

# res://components/movement_component.gd
class_name MovementComponent extends Node
@export var speed: float = 100.0
# ... movement logic ...

# res://enemies/goblin.tscn (Root: CharacterBody2D)
#   - HealthComponent
#   - MovementComponent
#   - Sprite2D (goblin_sprite.png)
#   - CollisionShape2D
# res://enemies/goblin.gd (attaches to goblin.tscn)
class_name Goblin extends CharacterBody2D
@onready var health_component: HealthComponent = $HealthComponent
@onready var movement_component: MovementComponent = $MovementComponent
# ... goblin specific logic using components ...
```

### 1.2. Single Responsibility Principle (SRP)
Each script and scene should have one, and only one, reason to change. This means a script should manage a single aspect of functionality (e.g., movement, health, inventory).

❌ **BAD:** God object script
```gdscript
# player.gd
extends CharacterBody2D
var health = 100
var speed = 150
func _physics_process(delta):
    # Movement logic
    # Attack logic
    # Inventory management
    # Health updates
```

✅ **GOOD:** Delegated responsibilities
```gdscript
# player.tscn (Root: CharacterBody2D)
#   - HealthComponent
#   - MovementComponent
#   - InventoryComponent
#   - AttackComponent
# player.gd (attaches to player.tscn)
extends CharacterBody2D
@onready var health_comp: HealthComponent = $HealthComponent
@onready var movement_comp: MovementComponent = $MovementComponent
@onready var inventory_comp: InventoryComponent = $InventoryComponent
@onready var attack_comp: AttackComponent = $AttackComponent

func _physics_process(delta):
    movement_comp.handle_input(self, delta) # Player handles input, passes to component
    attack_comp.update_cooldown(delta)
```

### 1.3. Autoloads for Global Services ONLY
Autoloads (Singletons) are for truly global, project-wide services like event buses, game state managers, or save/load systems. Avoid using them for scene-specific logic or as a general "dumping ground."

❌ **BAD:** Autoload for player data
```gdscript
# Autoload: PlayerData.gd
var player_health = 100
var player_position = Vector2.ZERO
# ... accessed globally from any scene ...
```

✅ **GOOD:** Autoload for event bus
```gdscript
# Autoload: EventBus.gd
extends Node
signal player_died
signal score_updated(new_score)
# ... other global events ...

# In Player.gd:
func take_damage(amount):
    health -= amount
    if health <= 0:
        EventBus.player_died.emit()

# In UI.gd:
func _ready():
    EventBus.player_died.connect(_on_player_died)
```

### 1.4. Consistent Naming Conventions
Follow Godot's official style guides for GDScript and C#. Consistency is key for readability and maintainability.

*   **GDScript (PEP 8 inspired):**
    *   `class_name`: `PascalCase`
    *   Variables, functions, signals: `snake_case`
    *   Constants: `SCREAMING_SNAKE_CASE`
    *   Private members (convention, not enforced): `_prefix_snake_case`

*   **C# (Microsoft conventions):**
    *   Namespaces, types, public members: `PascalCase`
    *   Local variables, method arguments: `camelCase`
    *   Private fields: `_camelCase` (with underscore prefix)

❌ **BAD:** Mixed GDScript style
```gdscript
var MyVariable = 10
func DoSomething(): pass
const MY_CONSTANT = 5
```

✅ **GOOD:** Consistent GDScript style
```gdscript
var my_variable = 10
func do_something(): pass
const MY_CONSTANT = 5
```

## 2. Common Patterns and Anti-patterns

### 2.1. Resource Loading: `preload()` vs. `load()`
Use `preload()` for resources needed at script load time (e.g., constants, exported PackedScenes). Use `load()` for dynamic, runtime loading, especially when the resource might change or be unloaded.

❌ **BAD:** `load()` for constant scenes
```gdscript
# In a script that's always loaded
var PlayerScene = load("res://scenes/player.tscn") # Loads every time script is instantiated
```

✅ **GOOD:** `preload()` for constant scenes
```gdscript
# In a script that's always loaded
const PLAYER_SCENE = preload("res://scenes/player.tscn") # Loads once when script object is loaded

@export var enemy_scene: PackedScene = preload("res://scenes/enemy.tscn") # Preload for editor convenience
```

### 2.2. Node Property Initialization
Always set a node's properties *before* adding it to the scene tree. This avoids unnecessary recalculations and potential performance hits.

❌ **BAD:** Setting properties after adding
```gdscript
var new_node = Node2D.new()
add_child(new_node)
new_node.position = Vector2(100, 50) # May trigger unnecessary updates
new_node.name = "MyNode"
```

✅ **GOOD:** Setting properties before adding
```gdscript
var new_node = Node2D.new()
new_node.position = Vector2(100, 50)
new_node.name = "MyNode"
add_child(new_node) # Add to tree only after initial setup
```

### 2.3. Signals for Loose Coupling
Use Godot's signal system to communicate between unrelated nodes or components. Avoid direct method calls between distant nodes (`get_node("../../../SomeNode").do_something()`) as this creates tight coupling.

❌ **BAD:** Direct node access
```gdscript
# Player.gd
func take_damage(amount):
    # ...
    get_node("/root/UI/HealthBar").update_health(health)
```

✅ **GOOD:** Using signals
```gdscript
# Player.gd
signal health_changed(new_health)
func take_damage(amount):
    health -= amount
    health_changed.emit(health)

# HealthBar.gd
func _ready():
    # Assuming Player is a sibling or accessible via a path
    var player = get_node("../Player")
    if player:
        player.health_changed.connect(_on_player_health_changed)

func _on_player_health_changed(new_health):
    update_health_display(new_health)
```

## 3. Performance Considerations

### 3.1. Avoid Heavy Operations in `_process` / `_physics_process`
These functions run every frame/physics frame. Offload complex calculations, file I/O, or network requests to separate threads, `await` for results, or run them less frequently.

❌ **BAD:** Complex raycasting every frame
```gdscript
func _process(delta):
    var space_state = get_world_2d().direct_space_state
    var query = PhysicsRayQueryParameters2D.create(...)
    var result = space_state.intersect_ray(query)
    # ... heavy processing ...
```

✅ **GOOD:** Event-driven or throttled operations
```gdscript
func _input(event):
    if event.is_action_pressed("interact"):
        _perform_interaction_raycast()

func _perform_interaction_raycast():
    var space_state = get_world_2d().direct_space_state
    var query = PhysicsRayQueryParameters2D.create(...)
    var result = space_state.intersect_ray(query)
    # ... process result ...
```

### 3.2. Dynamic Level Loading for Large Worlds
For large, open worlds, load and unload level sections dynamically. Don't load the entire world at once. Use `Area2D` or `CollisionShape3D` triggers to manage chunk loading/unloading.

```gdscript
# LevelChunkLoader.gd
extends Area2D
@export var chunk_scene: PackedScene
var loaded_chunk: Node = null

func _on_body_entered(body):
    if body.name == "Player" and loaded_chunk == null:
        loaded_chunk = chunk_scene.instantiate()
        get_parent().add_child(loaded_chunk)
        # Position chunk relative to loader

func _on_body_exited(body):
    if body.name == "Player" and loaded_chunk != null:
        loaded_chunk.queue_free()
        loaded_chunk = null
```

## 4. Common Pitfalls and Gotchas

### 4.1. Over-reliance on `get_node()`
Hardcoding node paths (`get_node("Path/To/Node")`) is fragile. If the scene structure changes, these paths break.

✅ **GOOD:** Use `@export` or `@onready` with direct references
```gdscript
# Use @export for editor assignment
@export var target_node: Node2D

# Use @onready for children/known paths
@onready var sprite: Sprite2D = $Sprite2D
@onready var health_component: HealthComponent = %HealthComponent # Use % for unique name in scene
```

### 4.2. Not Using `class_name`
`class_name` registers your script globally, allowing type-hinting, autocompletion, and easier referencing without `preload()`.

❌ **BAD:** Referencing scripts by path
```gdscript
# OtherScript.gd
var player_script = preload("res://player.gd")
var player_instance = player_script.new()
```

✅ **GOOD:** Using `class_name`
```gdscript
# player.gd
class_name Player extends CharacterBody2D

# OtherScript.gd
var player_instance = Player.new()
```

### 4.3. Ignoring GDScript Warnings
Godot's editor provides excellent warnings. Treat them as errors and fix them. They often point to potential bugs or performance issues.

## 5. Testing Approaches

### 5.1. Scene-Based Testing
Create dedicated test scenes for isolated components or specific game mechanics. This allows you to quickly iterate and verify functionality without running the entire game.

```gdscript
# res://tests/test_player_movement.tscn (Root: Node)
#   - Player (instantiated from res://scenes/player.tscn)
#   - Camera2D
#   - TestScript.gd (attached to root Node)

# test_player_movement.gd
extends Node
@onready var player = $Player

func _ready():
    # Simulate input, check player position, etc.
    print("Running player movement test...")
    player.position = Vector2.ZERO
    # Simulate a key press or call player's movement function directly
    # Assert player.position is as expected
    print("Player movement test complete.")
```

### 5.2. Unit Testing with GDUnit
For more rigorous unit testing of pure GDScript logic (e.g., utility functions, data structures), integrate a community-driven framework like GDUnit.

1.  **Install GDUnit:** Follow GDUnit's official installation guide.
2.  **Write Tests:**
    ```gdscript
    # res://tests/gdunit/test_math_utils.gd
    extends GDUnitTestSuite

    func test_add_numbers():
        assert_eq(2 + 2, 4)
        assert_eq(10 + 5, 15)

    func test_subtract_numbers():
        assert_eq(5 - 3, 2)
        assert_eq(10 - 20, -10)
    ```
3.  **Run Tests:** Use the GDUnit test runner scene.

### 5.3. Design for Testability
Adhere to SRP and loose coupling (using signals) to make your code easier to test. Components with clear inputs and outputs are simpler to isolate and verify. Avoid global state where possible.