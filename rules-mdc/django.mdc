---
description: Definitive guidelines for writing maintainable, performant, and secure Django applications, emphasizing modern best practices, clear code organization, and efficient patterns.
globs: **/*.py
---
# django Best Practices

This guide outlines the definitive best practices for developing Django applications, ensuring maintainability, performance, and security. Adhere to these rules for consistent, high-quality code.

## 1. Code Organization & Project Structure

Adopt the `src-layout` for clear separation of concerns. Keep apps focused on a single domain.

*   **Project Layout (`src-layout`)**:
    ```
    .
    ├── manage.py
    ├── src/
    │   ├── config/             # Project-level settings, URLs, WSGI/ASGI
    │   │   ├── __init__.py
    │   │   ├── settings/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── development.py
    │   │   │   └── production.py
    │   │   └── urls.py
    │   ├── apps/               # Domain-driven Django apps
    │   │   ├── users/
    │   │   │   ├── models.py
    │   │   │   ├── views.py
    │   │   │   └── tests/
    │   │   ├── products/
    │   │   └── ...
    │   └── common/             # Reusable utilities, abstract base models, etc.
    └── requirements.txt
    ```

*   **Split Settings**: Use `django-environ` for environment-specific settings. Never commit secrets.

    ❌ BAD: Hardcoding secrets, single `settings.py`
    ```python
    # config/settings.py
    DEBUG = True
    SECRET_KEY = 'super-secret-dev-key'
    DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': BASE_DIR / 'db.sqlite3'}}
    ```

    ✅ GOOD: Environment variables, split files
    ```python
    # src/config/settings/base.py
    import environ
    env = environ.Env()
    environ.Env.read_env() # reads .env file

    SECRET_KEY = env('SECRET_KEY')
    DEBUG = env.bool('DEBUG', default=False)
    ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])

    # src/config/settings/development.py
    from .base import *
    DEBUG = True
    DATABASES = {'default': env.db('DATABASE_URL', default='sqlite:///db.sqlite3')}

    # .env file (not committed to VCS)
    SECRET_KEY=your_actual_secret_key
    DEBUG=True
    DATABASE_URL=postgres://user:pass@host:port/dbname
    ```

*   **Model Naming**: Models are singular nouns. `related_name` for reverse relationships is plural.

    ❌ BAD:
    ```python
    class Users(models.Model): pass
    owner = models.ForeignKey(Owner, related_name='item')
    ```

    ✅ GOOD:
    ```python
    class User(models.Model): pass
    owner = models.ForeignKey(Owner, related_name='items', on_delete=models.CASCADE)
    ```

## 2. Common Patterns & Anti-patterns

*   **Fat Models, Skinny Views**: Business logic belongs in models or dedicated service layers, not views. Views orchestrate, models/services execute.

    ❌ BAD: Logic in view
    ```python
    # apps/orders/views.py
    def create_order_view(request):
        # ... validation ...
        product = Product.objects.get(id=product_id)
        if product.stock < quantity:
            raise ValidationError("Not enough stock")
        order = Order.objects.create(user=request.user, product=product, quantity=quantity)
        product.stock -= quantity
        product.save()
        # ...
    ```

    ✅ GOOD: Logic in model/service
    ```python
    # apps/orders/models.py
    class Order(models.Model):
        # ... fields ...
        @classmethod
        def create_with_stock_check(cls, user, product, quantity):
            if product.stock < quantity:
                raise ValidationError("Not enough stock")
            order = cls.objects.create(user=user, product=product, quantity=quantity)
            product.stock -= quantity
            product.save()
            return order

    # apps/orders/views.py
    def create_order_view(request):
        # ... validation ...
        order = Order.create_with_stock_check(request.user, product, quantity)
        # ...
    ```

## 3. Performance Considerations

*   **Optimize ORM Queries**: Avoid N+1 queries.

    ❌ BAD: N+1 query
    ```python
    # Iterates over users, then queries profile for each
    users = User.objects.all()
    for user in users:
        print(user.profile.bio)
    ```

    ✅ GOOD: `select_related` (one-to-one, foreign key)
    ```python
    users = User.objects.select_related('profile').all()
    for user in users:
        print(user.profile.bio)
    ```

    ✅ GOOD: `prefetch_related` (many-to-many, reverse foreign key)
    ```python
    # For a list of books, prefetch all authors for each book
    books = Book.objects.prefetch_related('authors').all()
    for book in books:
        print([author.name for author in book.authors.all()])
    ```

*   **Database Indexes**: Add `db_index=True` to frequently filtered/ordered fields.
    ```python
    class MyModel(models.Model):
        name = models.CharField(max_length=100, db_index=True) # Indexed
        created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    ```

*   **Async ORM**: Use `sync_to_async` for blocking ORM calls in async views, or the native async ORM (Django 4.1+).

    ```python
    # In an async view
    from asgiref.sync import sync_to_async

    async def my_async_view(request):
        # Blocking ORM call, wrap it
        user = await sync_to_async(User.objects.get)(id=request.user.id)
        # Or, if using native async ORM (Django 4.1+):
        # user = await User.objects.aget(id=request.user.id)
        return JsonResponse({'username': user.username})
    ```

## 4. Security Best Practices

*   **Secrets Management**: Never commit secrets to VCS. Use environment variables (see Split Settings).

*   **Permissions (DRF)**: Implement role-based permissions using DRF's `permission_classes`.

    ❌ BAD: Manual checks in view
    ```python
    class MyView(APIView):
        def get(self, request):
            if not request.user.is_staff:
                return Response(status=403)
            # ...
    ```

    ✅ GOOD: DRF Permission Classes
    ```python
    from rest_framework.permissions import IsAdminUser

    class MyView(APIView):
        permission_classes = [IsAdminUser] # Only staff can access
        def get(self, request):
            # ...
    ```

*   **Static & Media Files**: Serve static files with `ManifestStaticFilesStorage` and media files from cloud storage (e.g., S3, GCS).

    ```python
    # config/settings/production.py
    STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage' # e.g., django-storages
    ```

## 5. API Design (DRF)

*   **ViewSets & Routers**: Use `ModelViewSet` for CRUD operations to reduce boilerplate.

    ❌ BAD: Separate views for list and detail
    ```python
    class ProductListView(APIView): pass
    class ProductDetailView(APIView): pass
    # urls.py: path('products/', ProductListView.as_view()), path('products/<int:pk>/', ProductDetailView.as_view())
    ```

    ✅ GOOD: `ModelViewSet` with Router
    ```python
    # apps/products/views.py
    from rest_framework import viewsets
    from .models import Product
    from .serializers import ProductSerializer

    class ProductViewSet(viewsets.ModelViewSet):
        queryset = Product.objects.all()
        serializer_class = ProductSerializer

    # src/config/urls.py
    from rest_framework.routers import DefaultRouter
    from apps.products.views import ProductViewSet

    router = DefaultRouter()
    router.register(r'products', ProductViewSet)
    urlpatterns = [
        # ...
        path('api/', include(router.urls)),
    ]
    ```

## 6. Type Hints

Always use type hints for improved readability, maintainability, and static analysis with `mypy`.

❌ BAD: Untyped function
```python
def calculate_total(price, quantity):
    return price * quantity
```

✅ GOOD: Typed function
```python
def calculate_total(price: float, quantity: int) -> float:
    return price * quantity

# For models:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from django.db.models import QuerySet

class Product(models.Model):
    name: str = models.CharField(max_length=255)
    price: float = models.DecimalField(max_digits=10, decimal_places=2)

    def get_related_products(self) -> "QuerySet[Product]":
        # ...
        return Product.objects.filter(...)
```