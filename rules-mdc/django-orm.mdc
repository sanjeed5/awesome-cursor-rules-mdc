---
description: This guide provides opinionated, actionable best practices for writing efficient, maintainable, and performant Django ORM code, focusing on common pitfalls and modern patterns.
globs: **/*.py
---
# django-orm Best Practices

These rules are your definitive guide for writing robust and performant Django ORM code. Adhere to them strictly to ensure consistency, efficiency, and maintainability across our projects.

## 1. Model Structure and Naming Conventions

Always follow the official Django style guide for models. Consistency is paramount.

*   **Naming**:
    *   **Models**: `PascalCase` and singular.
    *   **Fields**: `snake_case`.
*   **Ordering**: Maintain a consistent order within your model definitions.

```python
# ❌ BAD: Inconsistent naming, poor ordering
from django.db import models

class books(models.Model):
    BookTitle = models.CharField(max_length=200)
    author_name = models.CharField(max_length=100)
    def __str__(self):
        return self.BookTitle
    class Meta:
        ordering = ['BookTitle']

# ✅ GOOD: Adheres to conventions
from django.db import models
from django.urls import reverse

class Book(models.Model):
    # 1. Choices (if any)
    class Status(models.TextChoices):
        DRAFT = "DR", "Draft"
        PUBLISHED = "PB", "Published"

    # 2. Database Fields
    title = models.CharField(max_length=200, verbose_name="Book Title")
    author = models.ForeignKey('Author', on_delete=models.CASCADE, related_name="books")
    status = models.CharField(max_length=2, choices=Status.choices, default=Status.DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)

    # 3. Custom Managers (if any)
    # objects = BookManager()

    # 4. Meta class
    class Meta:
        verbose_name = "book"
        verbose_name_plural = "books"
        ordering = ["-created_at", "title"] # Default ordering
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['author', 'status']),
        ]

    # 5. __str__ method
    def __str__(self):
        return f"{self.title} by {self.author.name}"

    # 6. save() method (if overridden)
    # def save(self, *args, **kwargs):
    #     super().save(*args, **kwargs)

    # 7. get_absolute_url() method
    def get_absolute_url(self):
        return reverse("book_detail", args=[str(self.id)])

    # 8. Custom methods (if any)
    def is_published(self):
        return self.status == self.Status.PUBLISHED
```

## 2. Explicit Field Definitions

Always be explicit with `null`, `blank`, `verbose_name`, and `related_name`. Avoid relying on Django's defaults for clarity and future maintainability.

*   **`null=True` vs `blank=True`**: `null` affects the database (can store `NULL`), `blank` affects form validation (can be empty). Use them correctly.
*   **`verbose_name`**: Provide human-readable names for fields.
*   **`related_name`**: Always define for `ForeignKey`, `ManyToManyField`, `OneToOneField` to avoid `_set` suffixes and prevent naming clashes. Use a plural, descriptive name.
*   **`choices`**: Use `TextChoices` or `IntegerChoices` for readability and type safety.

```python
# ❌ BAD: Implicit defaults, confusing related_name
class Product(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField() # blank=False, null=False implicitly
    category = models.ForeignKey('Category', on_delete=models.CASCADE) # related_name defaults to 'product_set'

# ✅ GOOD: Explicit, clear, and maintainable
from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=50, unique=True, verbose_name="Category Name")

    def __str__(self):
        return self.name

class Product(models.Model):
    class AvailabilityStatus(models.TextChoices):
        IN_STOCK = "IS", "In Stock"
        OUT_OF_STOCK = "OOS", "Out of Stock"
        PRE_ORDER = "PO", "Pre-Order"

    name = models.CharField(max_length=100, verbose_name="Product Name")
    description = models.TextField(blank=True, null=True, verbose_name="Product Description") # Can be empty in forms and DB
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="Unit Price")
    category = models.ForeignKey(
        Category,
        on_delete=models.PROTECT, # Prevent accidental deletion of categories with products
        related_name="products", # Explicit and clear reverse relation
        verbose_name="Product Category"
    )
    status = models.CharField(
        max_length=3,
        choices=AvailabilityStatus.choices,
        default=AvailabilityStatus.IN_STOCK,
        verbose_name="Availability Status"
    )

    def __str__(self):
        return self.name
```

## 3. Query Optimization: Avoid N+1 Queries

The N+1 query problem is a major performance killer. Always pre-fetch related data.

*   **`select_related()`**: For `ForeignKey` and `OneToOneField` relationships (single object joins). Fetches related objects in the *same* SQL query.
*   **`prefetch_related()`**: For `ManyToManyField` and reverse `ForeignKey` relationships (many-to-many or many-to-one). Performs a *separate* query for each related table and then joins them in Python.

```python
# ❌ BAD: N+1 queries
products = Product.objects.all()
for product in products:
    print(f"{product.name} ({product.category.name})") # Accessing product.category.name hits DB for EACH product

# ✅ GOOD: Single query for products and their categories
products = Product.objects.select_related('category').all()
for product in products:
    print(f"{product.name} ({product.category.name})") # Category is already loaded

# ❌ BAD: N+1 queries for many-to-many
book_authors = Book.objects.all()
for book in book_authors:
    authors = ", ".join([author.name for author in book.authors.all()]) # Hits DB for EACH book's authors

# ✅ GOOD: Efficiently prefetch many-to-many relationships
book_authors = Book.objects.prefetch_related('authors').all()
for book in book_authors:
    authors = ", ".join([author.name for author in book.authors.all()]) # Authors are pre-loaded
```

## 4. Efficient Data Retrieval

Retrieve only the data you need. Avoid hydrating full model instances when not necessary.

*   **`values()` / `values_list()`**: Use when you only need a subset of fields and don't require full model instances. `values_list(..., flat=True)` for single column lists.
*   **`exists()` / `count()`**: Use `exists()` when you only need to check if any records match a query, and `count()` when you need the total number. Avoid `len(queryset)` for counting.

```python
# ❌ BAD: Retrieves full model objects, then extracts titles
book_titles = [book.title for book in Book.objects.all()]

# ✅ GOOD: Retrieves only the 'title' field, more efficient
book_titles = Book.objects.values_list('title', flat=True)

# ❌ BAD: Inefficiently checks for existence
if len(Book.objects.filter(author__name="Jane Austen")) > 0:
    print("Jane Austen has books.")

# ✅ GOOD: Efficiently checks for existence
if Book.objects.filter(author__name="Jane Austen").exists():
    print("Jane Austen has books.")

# ❌ BAD: Retrieves all objects to count
total_books = len(Book.objects.all())

# ✅ GOOD: Performs a COUNT(*) query
total_books = Book.objects.count()
```

## 5. Database-Level Operations

Leverage the database for aggregations and atomic updates.

*   **`annotate()` / `aggregate()`**: Perform calculations (e.g., `Count`, `Sum`, `Avg`) directly in the database.
*   **`F()` expressions**: For atomic updates, preventing race conditions.

```python
from django.db.models import Count, Avg, F

# ❌ BAD: Python-level aggregation (N+1 queries or inefficient)
authors = Author.objects.all()
for author in authors:
    author.book_count = author.books.count() # N+1 query if not prefetch_related

# ✅ GOOD: Database-level aggregation
authors_with_book_counts = Author.objects.annotate(total_books=Count('books'))
for author in authors_with_book_counts:
    print(f"{author.name}: {author.total_books} books")

# ❌ BAD: Potential race condition for incrementing a counter
product = Product.objects.get(id=1)
product.stock += 1
product.save()

# ✅ GOOD: Atomic update using F() expression
Product.objects.filter(id=1).update(stock=F('stock') + 1)
```

## 6. Bulk Operations

For creating, updating, or deleting many objects, use bulk operations to minimize database round-trips.

```python
# ❌ BAD: Multiple database inserts
authors_data = [
    {"name": "Leo Tolstoy"},
    {"name": "Jane Austen"},
]
for data in authors_data:
    Author.objects.create(**data)

# ✅ GOOD: Single database insert for multiple objects
authors_to_create = [
    Author(name="Leo Tolstoy"),
    Author(name="Jane Austen"),
]
Author.objects.bulk_create(authors_to_create)

# ✅ GOOD: Bulk update
# Update all books published before 2000 to 'old' status
Book.objects.filter(published_date__year__lt=2000).update(status=Book.Status.DRAFT)
```

## 7. Indexing

Add database indexes to frequently filtered or ordered fields. Profile to determine actual benefits.

*   Use `db_index=True` on individual fields or `Meta.indexes` for multi-column indexes.

```python
# ❌ BAD: No index on a frequently filtered field
class Order(models.Model):
    customer = models.ForeignKey('Customer', on_delete=models.CASCADE)
    order_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20) # Frequently filtered by status

# ✅ GOOD: Add index for performance on common filters
class Order(models.Model):
    customer = models.ForeignKey('Customer', on_delete=models.CASCADE)
    order_date = models.DateTimeField(auto_now_add=True, db_index=True) # Index for date range queries
    status = models.CharField(max_length=20, db_index=True) # Index for status filtering

    class Meta:
        indexes = [
            models.Index(fields=['customer', 'order_date']), # Multi-column index
        ]
```

## 8. Asynchronous ORM (Django 5.x+)

For high-concurrency applications, leverage Django 5.x's native async ORM capabilities. Wrap synchronous ORM calls in `sync_to_async` if working in an async context with older Django versions or mixed codebases.

```python
# ❌ BAD: Blocking ORM call in an async view
import asyncio
from django.http import JsonResponse
from .models import Book

async def get_books_blocking(request):
    # This will block the event loop
    books = Book.objects.all()
    data = [{"title": book.title} for book in books]
    return JsonResponse(data, safe=False)

# ✅ GOOD: Native async ORM (Django 5.x+)
from django.http import JsonResponse
from .models import Book

async def get_books_async(request):
    books = await Book.objects.filter(status=Book.Status.PUBLISHED).values_list('title', flat=True)
    return JsonResponse(list(books), safe=False)

# ✅ GOOD: Using sync_to_async for older Django or mixed contexts
from asgiref.sync import sync_to_async
from django.http import JsonResponse
from .models import Book

async def get_books_sync_to_async(request):
    # Wrap synchronous ORM calls
    books = await sync_to_async(Book.objects.all)()
    data = await sync_to_async(lambda: [{"title": book.title} for book in books])()
    return JsonResponse(data, safe=False)
```

## 9. Debugging and Profiling

Always use the Django Debug Toolbar to inspect queries. It's your most valuable tool for identifying N+1 problems and slow queries.

## 10. Security

The ORM provides robust protection against SQL injection. Avoid raw SQL unless absolutely necessary, and always use parameterized queries if you do.

```python
# ❌ BAD: Vulnerable to SQL injection
from django.db import connection

def get_user_bad(username):
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM myapp_user WHERE username = '{username}'")
        return cursor.fetchone()

# ✅ GOOD: ORM handles sanitization
from .models import User

def get_user_good(username):
    return User.objects.filter(username=username).first()

# ✅ GOOD: Raw SQL with parameterized query (if ORM cannot express it)
from django.db import connection

def get_user_safe_raw_sql(username):
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM myapp_user WHERE username = %s", [username])
        return cursor.fetchone()
```