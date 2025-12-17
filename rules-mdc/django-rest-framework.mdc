---
description: Enforces modern, secure, and performant `django-rest-framework` best practices for API development, focusing on maintainability and scalability.
globs: **/*.py
---
# django-rest-framework Best Practices

This guide outlines our definitive standards for building APIs with Django REST Framework (DRF). Adhere to these rules to ensure consistency, security, and performance across all projects.

## 1. Code Organization and Structure

Always structure your API within a dedicated Django app, typically named `api`. Centralize DRF-specific settings and use `DefaultRouter` for URL management.

- **Dedicated `api/` app**: Isolate API logic.
- **`api_settings.py`**: Store all DRF global settings here, then import into `settings.py`.
- **`urls.py` with `DefaultRouter`**: Automate URL patterns for ViewSets.

```python
# project_root/api/apps.py
from django.apps import AppConfig

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

# project_root/api/settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema', # For OpenAPI
}

# project_root/settings.py
from .api.settings import REST_FRAMEWORK as API_REST_FRAMEWORK
REST_FRAMEWORK = API_REST_FRAMEWORK
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'rest_framework_simplejwt',
    'django_filters',
    'drf_spectacular', # For OpenAPI
    'api',
]

# project_root/api/urls.py
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, ProductViewSet
from django.urls import path, include

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')
router.register(r'products', ProductViewSet, basename='product')

urlpatterns = [
    path('', include(router.urls)),
]

# project_root/project_name/urls.py
from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('api.urls')), # Versioned API namespace
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]
```

## 2. Serializers: Efficiency and Validation

Prioritize `ModelSerializer` for CRUD operations. Use custom `Serializer` classes only for complex, non-model-backed data or advanced validation. Always use `select_related`/`prefetch_related` to prevent N+1 queries.

```python
from rest_framework import serializers
from django.db.models import QuerySet
from typing import Any

from .models import Product, Category, Review, User

# ❌ BAD: N+1 queries for related objects, exposing all fields
class BadProductSerializer(serializers.ModelSerializer):
    category = serializers.CharField(source='category.name')
    reviews = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'category', 'reviews']

    def get_reviews(self, obj: Product) -> list[dict[str, Any]]:
        return [{'text': r.text, 'rating': r.rating} for r in obj.reviews.all()]

# ✅ GOOD: Optimized queries, controlled field exposure, explicit types
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name']

class ReviewSerializer(serializers.ModelSerializer):
    user = serializers.SlugRelatedField(slug_field='username', read_only=True)

    class Meta:
        model = Review
        fields = ['id', 'user', 'text', 'rating', 'created_at']
        read_only_fields = ['user', 'created_at']

class ProductSerializer(serializers.ModelSerializer):
    category: CategorySerializer = CategorySerializer(read_only=True) # Type hint for nested serializer
    reviews: ReviewSerializer = ReviewSerializer(many=True, read_only=True) # Type hint for nested list

    class Meta:
        model = Product
        fields = ['id', 'name', 'price', 'category', 'reviews', 'stock_count']
        read_only_fields = ['stock_count'] # Fields that should not be updated via API

    def create(self, validated_data: dict[str, Any]) -> Product:
        # Custom logic if needed, e.g., setting owner
        validated_data['owner'] = self.context['request'].user
        return super().create(validated_data)

# ✅ GOOD: Custom serializer for non-model data or specific input validation
class UserRegistrationSerializer(serializers.Serializer):
    username: str = serializers.CharField(max_length=150)
    email: str = serializers.EmailField()
    password: str = serializers.CharField(write_only=True)
    password_confirm: str = serializers.CharField(write_only=True)

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({"password_confirm": "Passwords do not match."})
        return data

    def create(self, validated_data: dict[str, Any]) -> User:
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password']
        )
        return user
```

## 3. ViewSets & Routers: Streamlining Endpoints

Always use `ModelViewSet` for standard CRUD operations. For read-only resources, use `ReadOnlyModelViewSet`. Leverage `DefaultRouter` to automatically generate URLs.

```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.db.models import QuerySet
from typing import Any

from .models import Product, Category, User
from .serializers import ProductSerializer, CategorySerializer, UserRegistrationSerializer, ReviewSerializer

# ❌ BAD: Manual URL routing, repetitive logic for each HTTP method
# class ProductList(APIView):
#     def get(self, request): ...
#     def post(self, request): ...

# ✅ GOOD: ModelViewSet for full CRUD, optimized queryset, explicit permissions
class ProductViewSet(viewsets.ModelViewSet):
    queryset: QuerySet[Product] = Product.objects.select_related('category').prefetch_related('reviews__user').all()
    serializer_class: type[ProductSerializer] = ProductSerializer
    permission_classes: list[type[IsAuthenticated | IsAdminUser]] = [IsAuthenticated, IsAdminUser] # Example: Admin can manage products

    def get_queryset(self) -> QuerySet[Product]:
        # Optimize queryset for specific actions if needed
        if self.action == 'list':
            return self.queryset.filter(is_active=True)
        return self.queryset

    def perform_create(self, serializer: ProductSerializer) -> None:
        serializer.save(owner=self.request.user) # Automatically assign owner

    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def add_review(self, request: Any, pk: int | None = None) -> Response:
        product = self.get_object()
        serializer = ReviewSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save(product=product, user=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

# ✅ GOOD: ReadOnlyModelViewSet for read-only resources
class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset: QuerySet[Category] = Category.objects.all()
    serializer_class: type[CategorySerializer] = CategorySerializer
    permission_classes: list[type[IsAuthenticated]] = [IsAuthenticated]

# ✅ GOOD: Custom ViewSet for non-model operations (e.g., user registration)
class UserViewSet(viewsets.ViewSet):
    permission_classes: list[type[IsAuthenticated]] = [IsAuthenticated] # Adjust as needed for registration

    @action(detail=False, methods=['post'], permission_classes=[]) # No auth for registration
    def register(self, request: Any) -> Response:
        serializer = UserRegistrationSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response({"message": "User registered successfully."}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'])
    def me(self, request: Any) -> Response:
        serializer = ProductSerializer(request.user) # Example: reuse ProductSerializer for user details
        return Response(serializer.data)
```

## 4. Authentication & Permissions: Secure Access

Always use token-based authentication (JWT or DRF's `TokenAuthentication`) for stateless APIs. Implement granular permissions using DRF's built-in classes or custom ones. Never use `AllowAny` unless the endpoint is truly public.

```python
from rest_framework.permissions import BasePermission, SAFE_METHODS
from rest_framework.request import Request
from typing import Any

# ✅ GOOD: Custom object-level permission
class IsOwnerOrReadOnly(BasePermission):
    """
    Object-level permission to allow only owners of an object to edit it.
    Assumes the model instance has an `owner` attribute.
    """
    def has_object_permission(self, request: Request, view: Any, obj: Any) -> bool:
        if request.method in SAFE_METHODS:
            return True
        return obj.owner == request.user

# Apply to ViewSet:
# class ProductViewSet(viewsets.ModelViewSet):
#     permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
```

## 5. Pagination, Filtering & Versioning

Enable `PageNumberPagination` (default) or `CursorPagination` for large datasets. Use `django-filter` for query parameter filtering. Implement URL versioning (`/api/v1/`) for clear API evolution.

```python
# project_root/api/filters.py
import django_filters
from .models import Product

class ProductFilter(django_filters.FilterSet):
    min_price = django_filters.NumberFilter(field_name="price", lookup_expr='gte')
    max_price = django_filters.NumberFilter(field_name="price", lookup_expr='lte')
    category_name = django_filters.CharFilter(field_name="category__name", lookup_expr='icontains')

    class Meta:
        model = Product
        fields = ['category', 'is_active']

# Apply to ViewSet:
# class ProductViewSet(viewsets.ModelViewSet):
#     filterset_class = ProductFilter
```

## 6. Performance Considerations

Prevent N+1 queries by using `select_related` and `prefetch_related` in your `get_queryset` method.

```python
# ✅ GOOD: Optimized queryset in ViewSet
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    def get_queryset(self) -> QuerySet[Product]:
        # Always optimize for common list/retrieve actions
        return self.queryset.select_related('category').prefetch_related('reviews__user')
```

## 7. Error Handling

Leverage DRF's built-in exception handling. For custom errors, define a custom exception handler that returns consistent, machine-readable responses.

```python
# project_root/api/utils.py
from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
from typing import Any

def custom_exception_handler(exc: Exception, context: dict[str, Any]) -> Response | None:
    # Call DRF's default exception handler first
    response = exception_handler(exc, context)

    if response is not None:
        # Example: Customize validation error messages
        if response.status_code == status.HTTP_400_BAD_REQUEST and 'detail' not in response.data:
            response.data = {'errors': response.data}
        return response

    # Handle unhandled exceptions gracefully
    return Response(
        {'detail': 'An unexpected error occurred.'},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR
    )

# project_root/api/settings.py
REST_FRAMEWORK = {
    # ...
    'EXCEPTION_HANDLER': 'api.utils.custom_exception_handler',
}
```

## 8. Type Hints

Mandate type hints for all DRF components: serializers, views, viewsets, and custom permissions. This improves code readability, maintainability, and enables static analysis.

```python
# All examples above demonstrate mandatory type hints.
# Example:
# class ProductSerializer(serializers.ModelSerializer):
#     category: CategorySerializer = CategorySerializer(read_only=True)
#     def create(self, validated_data: dict[str, Any]) -> Product: ...
```

## 9. Rate Limiting

Implement DRF's throttling classes to protect your API from abuse.

```python
# project_root/api/settings.py
REST_FRAMEWORK = {
    # ...
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day',
        'burst': '60/min', # Custom throttle rate
    }
}

# Apply to ViewSet for custom throttling:
# class ProductViewSet(viewsets.ModelViewSet):
#     throttle_classes = [UserRateThrottle, AnonRateThrottle] # Or custom throttle
#     throttle_scope = 'burst' # Use the 'burst' rate defined in settings
```

## 10. Async Support

For high-throughput endpoints, isolate async code within `async def` views or specific `ViewSet` actions. Run your project with an ASGI server (Uvicorn/Daphne). DRF's native async support is evolving; consider `adrf` for full async ViewSets if truly necessary, but be aware it's still experimental.

```python
from rest_framework.views import APIView
from rest_framework.response import Response
import asyncio
import httpx # For async HTTP requests
from typing import Any

# ✅ GOOD: Isolate async logic to specific views/actions
class AsyncDataView(APIView):
    async def get(self, request: Any, *args: Any, **kwargs: Any) -> Response:
        # Simulate an async operation, e.g., calling an external async service
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/external-data")
            response.raise_for_status()
            data = response.json()
        return Response({"message": "Data fetched asynchronously", "data": data})

# project_root/api/urls.py
# from .views import AsyncDataView
# urlpatterns = [
#     path('async-data/', AsyncDataView.as_view()),
#     # ...
# ]
```