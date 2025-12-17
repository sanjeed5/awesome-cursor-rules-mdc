---
description: Definitive guidelines for writing clean, performant, and maintainable Jetpack Compose UI code, emphasizing best practices for component design, state management, and modifier usage.
globs: **/*
---
# Jetpack Compose Best Practices

Jetpack Compose is the future of Android UI. Adhering to these guidelines ensures your composables are readable, maintainable, performant, and scalable. This is your definitive guide for building high-quality Compose applications.

## 1. Code Organization & Naming

### 1.1 Name Composables in PascalCase
Composable functions that emit UI **must** start with an uppercase letter (PascalCase). This clearly distinguishes them from regular Kotlin functions.

❌ **BAD**
```kotlin
@Composable
fun myButton(text: String) { /* ... */ }
```

✅ **GOOD**
```kotlin
@Composable
fun MyButton(text: String) { /* ... */ }
```

### 1.2 Standardize Parameter Order
Always order parameters consistently. This improves readability and makes composable APIs predictable.

1.  `modifier: Modifier = Modifier`
2.  Required data parameters
3.  Optional UI-related styling parameters (e.g., `color`, `shape`, `style`)
4.  Callback lambdas (`onClick`, `onValueChange`)
5.  Trailing `@Composable` content lambdas

❌ **BAD**
```kotlin
@Composable
fun UserProfile(
    onClick: () -> Unit,
    user: User,
    modifier: Modifier = Modifier,
    backgroundColor: Color = MaterialTheme.colors.surface
) { /* ... */ }
```

✅ **GOOD**
```kotlin
@Composable
fun UserProfile(
    user: User,
    modifier: Modifier = Modifier,
    backgroundColor: Color = MaterialTheme.colors.surface,
    onClick: () -> Unit
) { /* ... */ }
```

### 1.3 Single Responsibility: Emit UI *or* Return a Value
A composable function **must** either emit UI content into the composition or return a value, but never both. If a composable needs to expose control surfaces, provide them as parameters.

❌ **BAD**
```kotlin
@Composable
fun MyTextWithId(text: String): String {
    Text(text)
    return "text_id_123" // Emits UI AND returns a value
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyText(text: String) {
    Text(text)
}

fun generateTextId(text: String): String {
    return "text_id_${text.hashCode()}" // Pure function, no UI emission
}
```

### 1.4 Emit Single Layout Node
A composable function **must** emit either 0 or 1 piece of layout. Avoid emitting multiple, disconnected layout nodes. This ensures composables are cohesive and behave predictably.

❌ **BAD**
```kotlin
@Composable
fun TwoTexts(text1: String, text2: String) {
    Text(text1)
    Text(text2) // Emits two root layout nodes
}
```

✅ **GOOD**
```kotlin
@Composable
fun TwoTexts(text1: String, text2: String) {
    Column { // Single root layout node
        Text(text1)
        Text(text2)
    }
}
```

## 2. Modifiers: The Contract

### 2.1 Accept and Respect a Single Modifier
Every UI-emitting composable **must** accept a `modifier: Modifier = Modifier` as its first parameter. This parameter **must** be passed to the composable's root UI-emitting child.

❌ **BAD**
```kotlin
@Composable
fun MyCard(content: @Composable () -> Unit) {
    Card { // No modifier parameter accepted
        content()
    }
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Card(modifier = modifier) { // Modifier passed to the root
        content()
    }
}
```

### 2.2 Avoid Internal Layout Modifiers
Composables **must not** apply layout-controlling modifiers (e.g., `padding()`, `fillMaxWidth()`, `width()`, `height()`, `size()`) to their root component internally. These **must** be provided by the caller via the `modifier` parameter to ensure reusability and flexibility.

❌ **BAD**
```kotlin
@Composable
fun ProfileImage(imageUrl: String) {
    Image(
        painter = rememberImagePainter(imageUrl),
        contentDescription = "Profile picture",
        modifier = Modifier.size(48.dp).padding(8.dp) // Layout modifiers applied internally
    )
}
```

✅ **GOOD**
```kotlin
@Composable
fun ProfileImage(imageUrl: String, modifier: Modifier = Modifier) {
    Image(
        painter = rememberImagePainter(imageUrl),
        contentDescription = "Profile picture",
        modifier = modifier // Caller controls size and padding
    )
}

// Usage:
ProfileImage(imageUrl = "...", modifier = Modifier.size(48.dp).padding(8.dp))
```

### 2.3 Do Not Reuse Modifier Instances
Each composable **must** receive its own `Modifier` instance. Reusing the same `Modifier` object across multiple composables or within a composable's internal hierarchy can lead to unpredictable behavior and unintended side effects.

❌ **BAD**
```kotlin
@Composable
fun ReusedModifierExample(modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text("Hello", modifier = modifier.padding(8.dp)) // Modifier reused
        Text("World", modifier = modifier.padding(8.dp)) // Modifier reused
    }
}
```

✅ **GOOD**
```kotlin
@Composable
fun ReusedModifierExample(modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text("Hello", modifier = Modifier.padding(8.dp)) // New modifier instance
        Text("World", modifier = Modifier.padding(8.dp)) // New modifier instance
    }
}
```

## 3. State Management & Architecture

### 3.1 Prefer Stateless & Controlled Composables
Design composables to be stateless and controlled. They **must** accept external state as parameters and expose events (callbacks) for state changes. This enables unidirectional data flow and simplifies testing.

❌ **BAD**
```kotlin
@Composable
fun MyToggle() {
    var isChecked by remember { mutableStateOf(false) } // State owned internally
    Switch(checked = isChecked, onCheckedChange = { isChecked = it })
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyToggle(
    isChecked: Boolean, // State hoisted
    onCheckedChange: (Boolean) -> Unit,
    modifier: Modifier = Modifier
) {
    Switch(checked = isChecked, onCheckedChange = onCheckedChange, modifier = modifier)
}
```

### 3.2 Hoist State Upwards
State **must** be hoisted to the lowest common ancestor that needs to read or modify it. This keeps most composables stateless, improving reusability and testability.

❌ **BAD**
```kotlin
@Composable
fun MyScreen(viewModel: MyViewModel) { // Passing ViewModel down
    val uiState = viewModel.uiState.collectAsState()
    MyContent(uiState.value.data, viewModel::onEvent) // Passing ViewModel functions
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyScreen(
    uiState: MyUiState, // Pass immutable data
    onAction: (MyAction) -> Unit // Pass specific event callbacks
) {
    MyContent(
        data = uiState.data,
        onItemClick = { item -> onAction(MyAction.ItemClicked(item)) }
    )
}
```

### 3.3 Build Small, Single-Purpose Composables
Decompose your UI into small, focused composables. Start with low-level building blocks and compose them into higher-level sections. This promotes reusability and maintainability.

❌ **BAD**
```kotlin
@Composable
fun BigComplexScreen() {
    Column {
        // All UI logic for header, list, footer in one composable
        Header()
        LazyColumn { /* ... */ }
        Footer()
    }
}
```

✅ **GOOD**
```kotlin
// Low-level building blocks
@Composable fun AppIcon(painter: Painter, contentDescription: String?, modifier: Modifier = Modifier) { /* ... */ }
@Composable fun BodyText(text: String, modifier: Modifier = Modifier) { /* ... */ }

// Higher-level component
@Composable
fun IconTextButton(
    title: String,
    icon: Painter,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    Surface(modifier = modifier.clickable(onClick = onClick)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            AppIcon(painter = icon, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            BodyText(text = title)
        }
    }
}

// Screen-level composition
@Composable
fun NavigationScreen(modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        IconTextButton(title = "Home", icon = painterResource(R.drawable.ic_home), onClick = { /* ... */ })
        IconTextButton(title = "Settings", icon = painterResource(R.drawable.ic_settings), onClick = { /* ... */ })
    }
}
```

### 3.4 Leverage Immutability for Stability
Define UI state using immutable data classes (`val` properties). This allows the Compose runtime to correctly identify changes and skip unnecessary recompositions, significantly improving performance.

❌ **BAD**
```kotlin
data class UserProfileState(var name: String, var email: String) // `var` makes it unstable
```

✅ **GOOD**
```kotlin
data class UserProfileState(val name: String, val email: String) // `val` makes it stable
```

## 4. Performance Considerations

### 4.1 Optimize Recomposition with `remember` and `derivedStateOf`
Use `remember` to cache values that don't change frequently across recompositions. Use `derivedStateOf` when a state is derived from other states and you want to avoid recomposing consumers unless the *derived* state actually changes.

❌ **BAD**
```kotlin
@Composable
fun MyListScreen(items: List<String>) {
    val filteredItems = items.filter { it.contains("A") } // Re-calculated on every recomposition
    LazyColumn {
        items(filteredItems) { Text(it) }
    }
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyListScreen(items: List<String>) {
    val filteredItems by remember(items) { // Only re-calculate if `items` changes
        derivedStateOf { items.filter { it.contains("A") } }
    }
    LazyColumn {
        items(filteredItems) { Text(it) }
    }
}
```

### 4.2 Use Lazy Containers for Large Lists
For lists with potentially many items, always use `LazyColumn`, `LazyRow`, or `LazyVerticalGrid`. These components only compose and lay out visible items, preventing performance bottlenecks.

❌ **BAD**
```kotlin
@Composable
fun MyScrollableList(items: List<String>) {
    Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
        items.forEach { Text(it) } // All items composed, even if not visible
    }
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyScrollableList(items: List<String>) {
    LazyColumn {
        items(items) { Text(it) } // Only visible items composed
    }
}
```

## 5. Theming & Consistency

### 5.1 Use MaterialTheme for Consistent Styling
Always wrap your application's UI in a `MaterialTheme` (or `MaterialTheme` 3) to ensure consistent typography, colors, and shapes across all components. Avoid creating custom styling objects outside of the theme system.

❌ **BAD**
```kotlin
@Composable
fun CustomButton(text: String, style: CustomButtonStyle) { /* ... */ }
```

✅ **GOOD**
```kotlin
@Composable
fun PrimaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Button(onClick = onClick, modifier = modifier, colors = ButtonDefaults.buttonColors(backgroundColor = MaterialTheme.colors.primary)) {
        Text(text = text, style = MaterialTheme.typography.button)
    }
}
```

### 5.2 Respect Scaffold Padding
When using `Scaffold`, always apply the provided `PaddingValues` to your main content. This ensures proper layout and avoids overlapping with top bars, bottom bars, or floating action buttons.

❌ **BAD**
```kotlin
@Composable
fun MyScreenContent() {
    Scaffold(topBar = { TopAppBar(title = { Text("Title") }) }) {
        Column { Text("Content") } // Ignores Scaffold padding
    }
}
```

✅ **GOOD**
```kotlin
@Composable
fun MyScreenContent() {
    Scaffold(topBar = { TopAppBar(title = { Text("Title") }) }) { paddingValues ->
        Column(modifier = Modifier.padding(paddingValues)) { // Applies Scaffold padding
            Text("Content")
        }
    }
}
```

## 6. Testing Approaches

### 6.1 Embrace Test-Driven Development (TDD)
Write UI tests for your composables *before* writing the implementation. This ensures components are well-defined, testable, and robust. Decompose your UI into small, testable units.

```kotlin
class FavoriteButtonTest {
    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun whenFavoriteIsTrueThenFilledHeartIconIsDisplayed() {
        composeTestRule.setContent {
            MyTheme {
                FavoriteButton(isFavorite = true, onClick = {})
            }
        }
        composeTestRule.onNodeWithContentDescription("Filled heart icon").assertIsDisplayed()
    }

    @Test
    fun whenFavoriteIsFalseThenEmptyHeartIconIsDisplayed() {
        composeTestRule.setContent {
            MyTheme {
                FavoriteButton(isFavorite = false, onClick = {})
            }
        }
        composeTestRule.onNodeWithContentDescription("Empty heart icon").assertIsDisplayed()
    }
}
```