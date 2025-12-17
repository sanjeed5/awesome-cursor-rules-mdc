---
description: Definitive guidelines for writing high-quality, performant, and maintainable Flutter applications using modern Dart 3.10+ practices.
globs: **/*
---
# flutter Best Practices

## Code Organization and Structure

### 1.1 Enforce Consistent Formatting and Linting
Always use `dart format` and enable `editor.formatOnSave` in your IDE. Integrate `flutter_lints` for robust static analysis. This eliminates style debates and catches common errors early.

❌ BAD (Manual formatting, inconsistent style)
```dart
// main.dart
void main() => runApp(MyApp());
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(title: 'My App', home: MyHomePage());
  }
}
```

✅ GOOD (Automated formatting, `flutter_lints` enabled)
```dart
// main.dart
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'My App',
      home: const MyHomePage(),
    );
  }
}
```

### 1.2 Adopt MVVM with Clear Layer Separation
Separate your application into distinct UI and Data layers. The UI layer consists of `Views` (widgets) and `ViewModels` (state logic). The Data layer comprises `Repositories` (business logic, data access) and `Services` (external API calls). This ensures maintainability and testability.

*   **`lib/` structure:**
    *   `features/`: Group by feature (e.g., `auth/`, `products/`).
        *   `auth/`:
            *   `presentation/`:
                *   `screens/`: `login_screen.dart` (View)
                *   `viewmodels/`: `login_viewmodel.dart` (ViewModel)
            *   `domain/`:
                *   `models/`: `user.dart` (Immutable data model)
                *   `repositories/`: `auth_repository.dart` (Abstract interface)
            *   `data/`:
                *   `repositories/`: `auth_repository_impl.dart` (Concrete implementation)
                *   `services/`: `auth_api_service.dart` (API calls)

### 1.3 Use Riverpod for State Management and Dependency Injection
Riverpod is the definitive choice for state management and dependency injection. It provides compile-time safety, simplifies testing, and eliminates common `Provider` pitfalls.

❌ BAD (InheritedWidget/Provider for DI, manual state updates)
```dart
// In a widget
final userRepo = Provider.of<UserRepository>(context);
// ...
userRepo.fetchUsers().then((_) => setState(() { /* ... */ }));
```

✅ GOOD (Riverpod for DI and state, auto-rebuilds)
```dart
// providers.dart
final authRepositoryProvider = Provider<AuthRepository>((ref) => AuthRepositoryImpl(ref.read(authApiService)));
final loginViewModelProvider = StateNotifierProvider<LoginViewModel, LoginState>((ref) => LoginViewModel(ref.read(authRepositoryProvider)));

// login_screen.dart (View)
class LoginScreen extends ConsumerWidget {
  const LoginScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final loginState = ref.watch(loginViewModelProvider);
    // ... UI with loginState.isLoading, loginState.error, etc.
    ElevatedButton(
      onPressed