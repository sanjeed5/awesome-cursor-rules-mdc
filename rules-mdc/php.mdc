---
description: This guide provides definitive best practices for writing clean, secure, and maintainable PHP code, strictly adhering to PSR-12 and modern PHP 8.4+ features.
globs: **/*
---
# PHP Best Practices

This document outlines the definitive best practices for PHP development within our team. Adherence to these guidelines ensures consistency, maintainability, and security across all projects.

## Critical Guidelines:

### 1. Code Organization and Structure (PSR-12 & Strict Types)

Always adhere to PSR-12 for coding style. This includes indentation, brace placement, and line length. Crucially, enable strict typing in every PHP file.

*   **Indentation**: Use 4 spaces, never tabs.
*   **Line Endings**: Unix LF only.
*   **Closing Tag**: Omit `?>` from files containing only PHP code.
*   **File Header Order**:
    1.  `<?php`
    2.  File-level docblock (optional)
    3.  `declare(strict_types=1);`
    4.  `namespace` declaration
    5.  `use` statements (class, then function, then constant, grouped)
    6.  The rest of the code.

❌ BAD:
```php
<?php
namespace App\Service;
use App\Model\User;
declare(strict_types=1);

class UserService {
    public function getUser($id) { // Missing return type, no strict types at top
        return User::find($id);
    }
}
```

✅ GOOD:
```php
<?php

declare(strict_types=1);

namespace App\Service;

use App\Model\User;
use App\Repository\UserRepository;

class UserService
{
    public function __construct(private UserRepository $userRepository) {}

    public function getUser(int $id): ?User
    {
        return $this->userRepository->find($id);
    }
}
```

### 2. Common Patterns and Anti-patterns

*   **Type Hinting**: Always use scalar and return type declarations. Leverage PHP 8+ union types, intersection types, and `mixed`/`static`/`never` where appropriate.
*   **Constructor Property Promotion**: Use for concise dependency injection.
*   **Match Expression**: Prefer `match` over `switch` for cleaner, more expressive comparisons.
*   **Null Coalescing Operator (`??`)**: Use to simplify checks for `null` or unset variables.
*   **Attributes**: Prefer native PHP attributes (`#[Attribute]`) over docblock annotations for metadata.

❌ BAD:
```php
class LegacyProcessor {
    public function process($type, $data) {
        switch ($type) { // Old switch, no types
            case 'foo': return 'Foo ' . $data;
            case 'bar': return 'Bar ' . $data;
            default: return null;
        }
    }
}
```

✅ GOOD:
```php
<?php

declare(strict_types=1);

#[Attribute(Attribute::TARGET_CLASS)]
class MyServiceAttribute {}

#[MyServiceAttribute]
class ModernProcessor
{
    public function process(string $type, string $data): ?string
    {
        return match ($type) {
            'foo' => 'Foo ' . $data,
            'bar' => 'Bar ' . $data,
            default => null,
        };
    }

    public function getSetting(array $config): string
    {
        // Use null coalescing for default values
        return $config['setting'] ?? 'default_value';
    }
}
```

### 3. Performance Considerations

*   **Latest PHP Version**: Always use the current stable PHP version (8.4+).
*   **Composer Autoloading**: Optimize Composer's autoloader for production (`composer dump-autoload --optimize --no-dev --classmap-authoritative`).
*   **Avoid N+1 Queries**: Eager load relationships in ORMs.
*   **Caching**: Implement application-level caching for frequently accessed data.

### 4. Common Pitfalls and Gotchas

*   **Loose Comparisons**: Always use strict comparison operators (`===`, `!==`).
*   **Error Suppression**: Never use the `@` operator. Handle errors gracefully with exceptions.
*   **Modifying Arrays During Iteration**: Leads to unpredictable behavior. Iterate over a copy or use appropriate collection methods.

### 5. Security Best Practices

*   **Input Validation & Sanitization**: Validate all external input. Use specific validation rules, not just `filter_var`.
*   **Output Escaping**: Escape all output based on context (HTML, URL, JavaScript, CSS). Use templating engines that auto-escape or dedicated functions (e.g., `htmlspecialchars`).
*   **Prepared Statements**: Always use parameterized queries (PDO or ORM) to prevent SQL injection.
*   **Password Hashing**: Use `password_hash()` with `PASSWORD_ARGON2ID` (or `PASSWORD_BCRYPT` as fallback). Never store plain text passwords.
*   **Dependency Updates**: Regularly update Composer dependencies and run `composer audit`.
*   **CSRF Protection**: Implement CSRF tokens for state-changing requests.

❌ BAD:
```php
$username = $_POST['username'];
$password = $_POST['password'];
$sql = "SELECT * FROM users WHERE username = '$username' AND password = '$password'"; // SQL Injection!
$db->query($sql);
echo "Hello, " . $_GET['name']; // XSS vulnerability!
```

✅ GOOD:
```php
<?php

declare(strict_types=1);

// Input Validation
if (!isset($_POST['username']) || !is_string($_POST['username'])) {
    throw new InvalidArgumentException('Invalid username.');
}
$username = trim($_POST['username']);

// Password Hashing
$hashedPassword = password_hash($_POST['password'], PASSWORD_ARGON2ID);

// Prepared Statements
$stmt = $db->prepare("SELECT * FROM users WHERE username = :username AND password = :password");
$stmt->execute([':username' => $username, ':password' => $hashedPassword]);

// Output Escaping
echo "Hello, " . htmlspecialchars($_GET['name'] ?? 'Guest', ENT_QUOTES, 'UTF-8');
```

### 6. Error Handling

*   **Exceptions**: Use exceptions for all exceptional conditions. Create custom exception classes for domain-specific errors.
*   **Centralized Handling**: Implement a global exception handler to catch unhandled exceptions and log them.
*   **Logging**: Use a robust logging library (e.g., Monolog) for all errors, warnings, and important events.

### 7. API Design

*   **RESTful Principles**: Design APIs to be resource-oriented, using standard HTTP methods (GET, POST, PUT, DELETE) and status codes.
*   **Consistent Naming**: Use consistent pluralized resource names (e.g., `/users`, not `/user`).
*   **Versioning**: Implement API versioning (e.g., `/v1/users`).
*   **Payload Validation**: Strictly validate all incoming request payloads.

### 8. Testing Approaches

*   **PHPUnit**: Write comprehensive unit and integration tests using PHPUnit.
*   **Test Doubles**: Use mocks, stubs, and fakes to isolate units of code during testing.
*   **Static Analysis**: Integrate PHPStan or Psalm into your CI/CD pipeline with a high strictness level.
*   **Continuous Integration**: Automate tests and static analysis checks on every commit.