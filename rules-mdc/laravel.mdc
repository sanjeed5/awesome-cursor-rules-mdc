---
description: Definitive guide for writing clean, performant, and secure Laravel applications, emphasizing modern best practices and common pitfalls.
globs: **/*.php
---
# Laravel Best Practices (2025)

This guide outlines the definitive best practices for developing robust, scalable, and maintainable Laravel applications. Adhere to these principles to ensure your codebase is clean, performant, and secure, aligning with PSR-12, PHP The Right Way, and Laravel's conventions.

## 1. Code Organization & Structure

### 1.1 Thin Controllers, Fat Models/Services

Controllers orchestrate requests; they do not contain business logic. Delegate complex operations to dedicated Service classes or Model methods.

❌ **BAD: Business logic in Controller**
```php
// app/Http/Controllers/OrderController.php
class OrderController extends Controller
{
    public function store(Request $request)
    {
        $order = new Order();
        $order->user_id = Auth::id();
        $order->total = $request->input('price') * 1.20; // Calculate total with tax
        $order->status = 'pending';
        $order->save();

        // ... more logic
    }
}
```

✅ **GOOD: Delegate to a Service**
```php
// app/Services/OrderService.php
class OrderService
{
    public function createOrder(array $data): Order
    {
        $order = new Order();
        $order->user_id = $data['user_id'];
        $order->total = $data['price'] * 1.20; // Business logic encapsulated
        $order->status = 'pending';
        $order->save();

        return $order;
    }
}

// app/Http/Controllers/OrderController.php
use App\Services\OrderService;

class OrderController extends Controller
{
    public function __construct(private OrderService $orderService) {}

    public function store(Request $request)
    {
        $order = $this->orderService->createOrder([
            'user_id' => Auth::id(),
            'price' => $request->input('price'),
        ]);

        return response()->json($order, 201);
    }
}
```

### 1.2 Centralized Validation with Form Requests

Always use Form Request classes for validation. This keeps controllers clean and validation logic reusable.

❌ **BAD: Validation in Controller**
```php
// app/Http/Controllers/PostController.php
class PostController extends Controller
{
    public function store(Request $request)
    {
        $request->validate([
            'title' => ['required', 'string', 'max:255'],
            'body' => ['required', 'string'],
        ]);
        // ...
    }
}
```

✅ **GOOD: Dedicated Form Request**
```php
// app/Http/Requests/StorePostRequest.php
class StorePostRequest extends FormRequest
{
    public function authorize(): bool { return true; } // Or implement authorization logic

    public function rules(): array
    {
        return [
            'title' => ['required', 'string', 'max:255'],
            'body' => ['required', 'string'],
        ];
    }
}

// app/Http/Controllers/PostController.php
use App\Http\Requests\StorePostRequest;

class PostController extends Controller
{
    public function store(StorePostRequest $request)
    {
        // Validated data is automatically available
        $post = Post::create($request->validated());
        return response()->json($post, 201);
    }
}
```

## 2. Common Patterns & Anti-patterns

### 2.1 Eloquent Over Raw Queries (and Eager Loading)

Prefer Eloquent ORM for database interactions. Always eager load relationships to prevent N+1 query problems.

❌ **BAD: Raw queries & N+1 problem**
```php
// In a controller or view
$users = DB::select('SELECT * FROM users');
foreach ($users as $user) {
    echo $user->profile->bio; // N+1 query for each profile
}
```

✅ **GOOD: Eloquent with eager loading**
```php
// In a controller
$users = User::with('profile')->get();
// In a Blade template (data passed from controller)
@foreach ($users as $user)
    {{ $user->profile->bio }}
@endforeach
```

### 2.2 Collections Over Arrays

Leverage Laravel's powerful Collection class for data manipulation.

❌ **BAD: Manual array manipulation**
```php
$users = User::all()->toArray();
$activeUsers = [];
foreach ($users as $user) {
    if ($user['is_active']) {
        $activeUsers[] = $user;
    }
}
```

✅ **GOOD: Use Laravel Collections**
```php
$activeUsers = User::all()->filter(fn ($user) => $user->is_active);
// Or chain methods:
$activeUserNames = User::all()
    ->filter(fn ($user) => $user->is_active)
    ->map(fn ($user) => $user->name);
```

### 2.3 Mass Assignment Protection

Always define `$fillable` or `$guarded` properties on your Eloquent models to prevent mass assignment vulnerabilities.

❌ **BAD: No mass assignment protection**
```php
// app/Models/User.php - Missing $fillable or $guarded
class User extends Model {}

// In a controller, allows any request field to be saved
User::create($request->all());
```

✅ **GOOD: Define `$fillable`**
```php
// app/Models/User.php
class User extends Model
{
    protected $fillable = ['name', 'email', 'password'];
}

// In a controller, only specified fields can be mass assigned
User::create($request->validated()); // Assuming validated() returns fillable fields
```

### 2.4 No Database Queries in Blade Templates

Blade templates are for presentation. Pass all necessary data from the controller.

❌ **BAD: Querying in Blade**
```blade
<!-- resources/views/product.blade.php -->
<div>
    <h1>{{ $product->name }}</h1>
    @foreach (App\Models\Review::where('product_id', $product->id)->get() as $review)
        <p>{{ $review->comment }}</p>
    @endforeach
</div>
```

✅ **GOOD: Pass pre-loaded data**
```php
// app/Http/Controllers/ProductController.php
class ProductController extends Controller
{
    public function show(Product $product)
    {
        $product->load('reviews'); // Eager load reviews
        return view('product', compact('product'));
    }
}
```

## 3. Performance Considerations

### 3.1 Chunking Large Datasets

When processing large numbers of records, use `chunk()` or `chunkById()` to avoid memory exhaustion.

❌ **BAD: Loading all records into memory**
```php
User::where('active', true)->get()->each(function (User $user) {
    // Process each user
});
```

✅ **GOOD: Process in chunks**
```php
User::where('active', true)->chunk(1000, function (Collection $users) {
    foreach ($users as $user) {
        // Process each user in batches
    }
});
```

### 3.2 Caching Expensive Operations

Cache results of expensive queries or computations using Laravel's cache facade.

❌ **BAD: Repeated expensive queries**
```php
// Every request hits the database for all posts
$posts = Post::with('author')->get();
```

✅ **GOOD: Cache with a TTL**
```php
$posts = Cache::remember('all_posts_with_authors', 60 * 60, function () {
    return Post::with('author')->get();
});
```

### 3.3 Queue Background Jobs

Offload long-running tasks (e.g., sending emails, image processing, API calls) to queues.

❌ **BAD: Blocking user requests**
```php
// In a controller, sends email during HTTP request
Mail::to($user->email)->send(new WelcomeEmail($user));
```

✅ **GOOD: Dispatch to queue**
```php
// In a controller, dispatches email to a background queue
Mail::to($user->email)->send(new WelcomeEmail($user))->onQueue('emails');
```

## 4. Security Best Practices

### 4.1 Enforce HTTPS + HSTS

Always deploy with HTTPS and enable HSTS (HTTP Strict Transport Security) to prevent downgrade attacks. Configure this at your web server (Nginx/Apache) or load balancer.

### 4.2 Authentication & Authorization

*   **Authentication:** Use **Sanctum** for first-party SPAs and mobile apps. Use **Passport** if you need full OAuth2 support for third-party applications.
*   **Authorization:** Implement **Policies** for model-specific authorization (e.g., "Can this user update this post?"). Use **Gates** for more general permissions (e.g., "Can this user access the admin dashboard?").

❌ **BAD: Manual authorization checks**
```php
// In a controller
if (Auth::user()->role !== 'admin' || Auth::user()->id !== $post->user_id) {
    abort(403);
}
```

✅ **GOOD: Use Policies**
```php
// app/Policies/PostPolicy.php
class PostPolicy
{
    public function update(User $user, Post $post): bool
    {
        return $user->id === $post->user_id;
    }
}

// app/Http/Controllers/PostController.php
class PostController extends Controller
{
    public function update(Request $request, Post $post)
    {
        $this->authorize('update', $post); // Automatically checks policy
        // ...
    }
}
```

### 4.3 Rate Limiting

Apply rate limiting to API endpoints and sensitive routes to prevent abuse and brute-force attacks.

❌ **BAD: No rate limiting on API**
```php
Route::post('/api/register', [AuthController::class, 'register']);
```

✅ **GOOD: Apply `throttle` middleware**
```php
Route::middleware('throttle:60,1')->group(function () {
    Route::post('/api/register', [AuthController::class, 'register']);
});
```

### 4.4 Environment Configuration

Never leave `APP_DEBUG=true` in production. Store sensitive credentials in `.env` and access them via `config()`.

❌ **BAD: Direct `env()` access & debug in prod**
```php
// In code
$apiKey = env('STRIPE_KEY'); // Should be config()
// .env in production: APP_DEBUG=true
```

✅ **GOOD: Use `config()` and secure `.env`**
```php
// config/services.php
'stripe' => [
    'key' => env('STRIPE_KEY'),
],

// In code
$apiKey = config('services.stripe.key');
// .env in production: APP_DEBUG=false
```

## 5. API Design

### 5.1 URL-Based Versioning

Implement URL-based versioning for your APIs (e.g., `/api/v1/users`). This is explicit, cache-friendly, and easy for developers to understand.

❌ **BAD: No versioning or header-based**
```php
Route::get('/api/users', [UserController::class, 'index']); // Breaking changes will affect all clients
```

✅ **GOOD: URL-based versioning**
```php
Route::prefix('v1')->group(function () {
    Route::apiResource('users', UserController::class);
    // ...
});
```

## 6. Testing Approaches

### 6.1 Prioritize Feature Tests

Write feature tests using PHPUnit or Pest to cover critical user flows and API interactions. This ensures your application behaves as expected from an end-to-end perspective.

```php
// tests/Feature/UserRegistrationTest.php
class UserRegistrationTest extends TestCase
{
    use RefreshDatabase;

    /** @test */
    public function a_user_can_register()
    {
        $response = $this->postJson('/api/v1/register', [
            'name' => 'John Doe',
            'email' => 'john@example.com',
            'password' => 'password',
            'password_confirmation' => 'password',
        ]);

        $response->assertStatus(201)
                 ->assertJson(['message' => 'Registration successful']);

        $this->assertDatabaseHas('users', ['email' => 'john@example.com']);
    }
}
```