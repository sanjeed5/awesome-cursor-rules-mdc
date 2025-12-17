---
description: Definitive guidelines for using Riverpod in Flutter projects, focusing on modern, opinionated best practices for state management, code organization, performance, and testing.
globs: **/*
---
# riverpod Best Practices

Riverpod is the definitive state management solution for modern Flutter applications. It provides compile-time safety, unparalleled testability, and a declarative API. This guide outlines our team's mandatory best practices for leveraging Riverpod effectively.

## 1. Code Organization and Structure

Organize providers by feature or domain, not by provider type. This enhances modularity and discoverability. Always use `@riverpod` for code generation.

### ✅ GOOD: Feature-based Provider Grouping

```dart
// features/auth/auth_providers.dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:your_app/features/auth/auth_service.dart';

part 'auth_providers.g.dart';

@riverpod
AuthService authService(AuthServiceRef ref) {
  return AuthServiceImpl(); // Or mock in tests
}

@riverpod
class AuthNotifier extends _$AuthNotifier {
  @override
  bool build() {
    // Initial auth state logic
    return false;
  }

  void login() => state = true;
  void logout() => state = false;
}
```

### ❌ BAD: Generic or Type-based Grouping

```dart
// providers/auth_provider.dart (too generic)
// providers/future_providers.dart (grouped by type, not feature)
final authServiceProvider = Provider<AuthService>((ref) => AuthServiceImpl());
final authNotifierProvider = StateNotifierProvider<AuthNotifier, bool>((ref) => AuthNotifier());
```

**Action**: Always run `dart run build_runner watch` during development to keep `.g.dart` files up-to-date.

## 2. Provider Types and Usage

Prefer generated `@riverpod` classes (`Notifier`, `AsyncNotifier`, `StreamNotifier`) for mutable state and `FutureProvider`/`StreamProvider` for immutable async data.

### 2.1. Mutable State: `Notifier` and `AsyncNotifier`

Use `Notifier` for synchronous mutable state and `AsyncNotifier` for asynchronous mutable state (e.g., API calls that modify state).

### ✅ GOOD: `AsyncNotifier` for state with async operations

```dart
// features/todos/todo_providers.dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:your_app/features/todos/todo_repository.dart';
import 'package:your_app/features/todos/todo_model.dart';

part 'todo_providers.g.dart';

@riverpod
class TodoListNotifier extends _$TodoListNotifier {
  @override
  Future<List<Todo>> build() async {
    // Initial fetch
    return ref.watch(todoRepositoryProvider).fetchTodos();
  }

  Future<void> addTodo(Todo todo) async {
    state = const AsyncValue.loading(); // Show loading state
    state = await AsyncValue.guard(() async {
      final newTodo = await ref.read(todoRepositoryProvider).addTodo(todo);
      return [...state.value!, newTodo]; // Update state immutably
    });
  }
}
```

### ❌ BAD: `StateProvider` for complex or async mutable state

```dart
// This quickly becomes unmanageable for complex logic or async operations
final todoListProvider = StateProvider<List<Todo>>((ref) => []);
// How to handle loading/error for addTodo? Manual boilerplate.
```

### 2.2. Immutable Async Data: `FutureProvider` and `StreamProvider`

Use `FutureProvider` for one-off asynchronous data fetches and `StreamProvider` for real-time data streams.

### ✅ GOOD: `FutureProvider` for fetching data

```dart
// features/products/product_providers.dart
import 'package:riverpod_annotation/riverpod_annotation.dart';
import 'package:your_app/features/products/product_repository.dart';
import 'package:your_app/features/products/product_model.dart';

part 'product_providers.g.dart';

@riverpod
Future<List<Product>> products(ProductsRef ref) async {
  return ref.watch(productRepositoryProvider).getProducts();
}

// Parameterized provider using .family
@riverpod
Future<Product> product(ProductRef ref, String productId) async {
  return ref.watch(productRepositoryProvider).getProductById(productId);
}
```

### ❌ BAD: Manual `FutureBuilder` or `StreamBuilder`

```dart
// Avoid this boilerplate in UI when Riverpod can manage it
class ProductScreen extends StatelessWidget {
  final String productId;
  ProductScreen(this.productId);

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<Product>(
      future: ProductRepository().getProductById(productId), // No caching, no auto-dispose
      builder: (context, snapshot) {
        // Manual loading/error handling
        if (snapshot.connectionState == ConnectionState.waiting) {
          return CircularProgressIndicator();
        } else if (snapshot.hasError) {
          return Text('Error: ${snapshot.error}');
        } else if (snapshot.hasData) {
          return Text(snapshot.data!.name);
        }
        return SizedBox.shrink();
      },
    );
  }
}
```

## 3. `ref.watch`, `ref.read`, `ref.listen`

Understand the distinct purposes of these methods to control reactivity and side effects.

### 3.1. `ref.watch`: Reactive UI Updates

Use `ref.watch` in `build` methods of `ConsumerWidget` or `ConsumerStatefulWidget` to rebuild the UI when a provider's state changes.

### ✅ GOOD: Watching state for UI updates

```dart
class HomeScreen extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAuthenticated = ref.watch(authNotifierProvider); // UI rebuilds on auth state change
    final todosAsync = ref.watch(todoListNotifierProvider); // UI rebuilds on todo list change

    return todosAsync.when(
      data: (todos) => ListView.builder(
        itemCount: todos.length,
        itemBuilder: (context, index) => Text(todos[index].title),
      ),
      loading: () => const CircularProgressIndicator(),
      error: (err, stack) => Text('Error: $err'),
    );
  }
}
```

### ❌ BAD: Using `ref.read` for reactive UI

```dart
class HomeScreen extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // This will NOT rebuild the UI when authNotifierProvider changes!
    final isAuthenticated = ref.read(authNotifierProvider);
    return Text('User is authenticated: $isAuthenticated');
  }
}
```

### 3.2. `ref.read`: One-off Actions

Use `ref.read` for triggering actions (e.g., button presses, form submissions) that do not require the UI to react to the provider's state changes.

### ✅ GOOD: Reading for actions

```dart
class LoginButton extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return ElevatedButton(
      onPressed: () {
        // Triggers login without causing this widget to rebuild on auth state change
        ref.read(authNotifierProvider.notifier).login();
      },
      child: const Text('Login'),
    );
  }
}
```

### ❌ BAD: Watching for one-off actions

```dart
class LoginButton extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // This watches the provider, causing unnecessary rebuilds if the provider's state changes
    // even if the button itself doesn't need to react.
    final authNotifier = ref.watch(authNotifierProvider.notifier);
    return ElevatedButton(
      onPressed: () {
        authNotifier.login();
      },
      child: const Text('Login'),
    );
  }
}
```

### 3.3. `ref.listen`: Side Effects

Use `ref.listen` for side effects like showing snackbars, navigating, or logging, which should not cause UI rebuilds.

### ✅ GOOD: Listening for side effects

```dart
class AuthChecker extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    ref.listen<bool>(authNotifierProvider, (previous, next) {
      if (next == true) {
        // Navigate to home screen
        Navigator.of(context).pushReplacementNamed('/home');
      } else {
        // Navigate to login screen
        Navigator.of(context).pushReplacementNamed('/login');
      }
    });
    return const SizedBox.shrink(); // This widget doesn't render anything
  }
}
```

### ❌ BAD: Performing side effects in `build` or `ref.watch`

```dart
class AuthChecker extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final isAuthenticated = ref.watch(authNotifierProvider);
    // This is problematic:
    // 1. Causes navigation on every rebuild if isAuthenticated is true.
    // 2. Can lead to "setState during build" errors.
    if (isAuthenticated) {
      Navigator.of(context).pushReplacementNamed('/home');
    } else {
      Navigator.of(context).pushReplacementNamed('/login');
    }
    return const SizedBox.shrink();
  }
}
```

## 4. Performance Considerations

Optimize rebuilds and resource management.

### 4.1. `autoDispose` for Transient State

Always use `.autoDispose` for providers whose state is only needed temporarily (e.g., page-specific controllers, search results).

### ✅ GOOD: Auto-disposing temporary state

```dart
@riverpod
@Riverpod(keepAlive: false) // Explicitly autoDispose
Future<List<SearchResult>> searchResults(SearchResultsRef ref, String query) async {
  // Fetches search results, automatically disposed when not watched
  return ref.watch(searchRepositoryProvider).search(query);
}
```

### ❌ BAD: Not disposing temporary state

```dart
@riverpod // Defaults to keepAlive: true if not specified in older versions or without generator
Future<List<SearchResult>> searchResults(SearchResultsRef ref, String query) async {
  // This provider's state will persist in memory even if no longer used.
  return ref.watch(searchRepositoryProvider).search(query);
}
```

### 4.2. Selectors for Fine-Grained Rebuilds

Use `.select` to watch only specific parts of a provider's state, preventing unnecessary widget rebuilds.

### ✅ GOOD: Using selectors

```dart
class UserProfileHeader extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Only rebuilds if the user's name changes, not if other user properties change
    final userName = ref.watch(userNotifierProvider.select((user) => user.name));
    return Text('Welcome, $userName!');
  }
}
```

### ❌ BAD: Watching entire object for a single property

```dart
class UserProfileHeader extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Rebuilds if *any* property of the user object changes, even if only name is used.
    final user = ref.watch(userNotifierProvider);
    return Text('Welcome, ${user.name}!');
  }
}
```

## 5. Common Pitfalls and Gotchas

### 5.1. Forgetting `build_runner`

Always run `dart run build_runner watch` or `build_runner build` after modifying `@riverpod` annotated files.

### ❌ BAD: Missing `.g.dart` file

```dart
// auth_providers.dart
@riverpod
class AuthNotifier extends _$AuthNotifier { /* ... */ }

// main.dart
// Error: 'authNotifierProvider' is not defined.
// This happens if auth_providers.g.dart is not generated or imported.
final authState = ref.watch(authNotifierProvider);
```

### ✅ GOOD: Generated and imported `.g.dart`

```dart
// auth_providers.dart
part 'auth_providers.g.dart'; // Ensure this line is present
@riverpod
class AuthNotifier extends _$AuthNotifier { /* ... */ }

// main.dart
import 'package:your_app/features/auth/auth_providers.dart'; // Imports the generated provider
final authState = ref.watch(authNotifierProvider);
```

### 5.2. Mutable State Outside Notifiers

Always ensure state exposed by providers is immutable. Mutations should only occur within `Notifier` or `AsyncNotifier` classes.

### ❌ BAD: Directly modifying state from UI

```dart
// In a widget:
ElevatedButton(
  onPressed: () {
    // This is an anti-pattern: directly modifying a list watched by a provider
    ref.read(todoListNotifierProvider.notifier).state.add(new Todo(...));
  },
  child: const Text('Add Todo'),
);
```

### ✅ GOOD: Immutable state updates via Notifier methods

```dart
// In TodoListNotifier:
void addTodo(Todo todo) {
  state = AsyncValue.data([...state.value!, todo]); // Creates a new list
}

// In a widget:
ElevatedButton(
  onPressed: () {
    ref.read(todoListNotifierProvider.notifier).addTodo(new Todo(...));
  },
  child: const Text('Add Todo'),
);
```

## 6. Testing Approaches

Riverpod makes testing straightforward by allowing providers to be overridden.

### ✅ GOOD: Overriding providers in tests

```dart
// Mock implementation
class MockAuthService implements AuthService {
  @override
  Future<bool> login(String email, String password) async => true;
  @override
  Future<void> logout() async {}
}

void main() {
  testWidgets('Login screen shows home after successful login', (tester) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          // Override the real AuthService with a mock
          authServiceProvider.overrideWithValue(MockAuthService()),
          // Optionally override the notifier's initial state
          authNotifierProvider.overrideWith((ref) => AuthNotifier()..state = false),
        ],
        child: MaterialApp(
          home: LoginScreen(),
          routes: {'/home': (context) => HomeScreen()},
        ),
      ),
    );

    // ... perform login actions and verify navigation
  });
}
```

### ❌ BAD: Relying on real services in tests

```dart
// This test will hit actual network requests or database, making it slow and flaky.