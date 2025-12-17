---
description: Definitive guidelines for using Tortoise ORM effectively, focusing on async patterns, robust model design, efficient querying, and production-ready migrations.
globs: **/*.py
---
# tortoise-orm Best Practices

Tortoise ORM is an excellent choice for `asyncio`-native Python applications. Adhere to these guidelines for maintainable, performant, and secure database interactions.

## 1. Async Lifecycle Management

Always initialize and close connections explicitly. Failing to do so leads to resource leaks and unpredictable behavior.

❌ BAD: Dangling connections
```python
from tortoise import Tortoise

async def main():
    await Tortoise.init(db_url='sqlite://:memory:', modules={'models': ['app.models']})
    # ... application logic ...
    # No close_connections()
```

✅ GOOD: Proper async context management
```python
from tortoise import Tortoise, run_async

async def init_db():
    await Tortoise.init(
        db_url='sqlite://:memory:',
        modules={'models': ['app.models']}
    )
    await Tortoise.generate_schemas() # For development only

async def close_db():
    await Tortoise.close_connections()

# Use in your application entry point:
async def run_app():
    await init_db()
    try:
        # Your application code here
        pass
    finally:
        await close_db()

# Or, for simple scripts:
run_async(run_app())
```

## 2. Production Migrations with Aerich

`Tortoise.generate_schemas()` is for development convenience only. For production, use Aerich, the official migration tool.

❌ BAD: Using `generate_schemas` in production
```python
# app.py
await Tortoise.init(...)
await Tortoise.generate_schemas() # 🚨 DANGER: Do NOT use in production!
```

✅ GOOD: Aerich for schema evolution
1.  **Install Aerich:** `pip install "aerich[toml]"`
2.  **Configure `TORTOISE_ORM` to include `aerich.models`:**
    ```python
    # config.py
    TORTOISE_ORM = {
        "connections": {"default": "postgres://user:pass@host:port/db"},
        "apps": {
            "models": {
                "models": ["app.models", "aerich.models"], # Crucial: include aerich.models
                "default_connection": "default",
            },
        },
    }
    ```
3.  **Initialize Aerich:** `aerich init -t config.TORTOISE_ORM`
4.  **Generate initial migration:** `aerich init-db`
5.  **Generate subsequent migrations:** `aerich migrate`
6.  **Apply migrations:** `aerich upgrade`

## 3. Robust Model Design

Define models clearly with explicit types, primary keys, and `Meta` options.

### 3.1 Base Model & Primary Keys
Every model must inherit from `tortoise.models.Model` and define a primary key. Use recommended primary key types.

❌ BAD: Implicit primary key, missing type hints
```python
from tortoise.models import Model
from tortoise import fields

class User(Model):
    name = fields.TextField() # 'id' will be auto-generated, but less explicit
```

✅ GOOD: Explicit primary key, type hints
```python
from tortoise.models import Model
from tortoise import fields
from uuid import UUID

class User(Model):
    id: UUID = fields.UUIDField(primary_key=True) # Explicit UUID PK
    name: str = fields.CharField(max_length=255, unique=True)
    created_at: fields.DatetimeField = fields.DatetimeField(auto_now_add=True)
    updated_at: fields.DatetimeField = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "users" # Explicit table name
```

### 3.2 Abstract Models for Reusability
Use abstract models for common fields to avoid repetition.

```python
from tortoise.models import Model
from tortoise import fields

class TimestampMixin(Model):
    created_at = fields.DatetimeField(null=True, auto_now_add=True)
    updated_at = fields.DatetimeField(null=True, auto_now=True)

    class Meta:
        abstract = True # This model will not create a table

class Product(TimestampMixin):
    id: int = fields.IntField(primary_key=True)
    name: str = fields.CharField(max_length=100)
    price: float = fields.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        table = "products"
```

### 3.3 Model Discovery (`__models__`)
For explicit model discovery, define `__models__` in your model modules.

```python
# app/models.py
from tortoise.models import Model
from tortoise import fields

class Category(Model):
    id: int = fields.IntField(primary_key=True)
    name: str = fields.CharField(max_length=50, unique=True)

class Item(Model):
    id: int = fields.IntField(primary_key=True)
    name: str = fields.CharField(max_length=100)
    category: fields.ForeignKeyRelation["Category"] = fields.ForeignKeyField('models.Category', related_name='items')

__models__ = [
    "app.models.Category",
    "app.models.Item",
]
```

## 4. Efficient Querying (Avoid N+1)

Always eager-load related objects to prevent the N+1 query problem.

❌ BAD: N+1 queries
```python
# Fetching each category's items individually
categories = await Category.all()
for category in categories:
    print(f"Category: {category.name}")
    # This triggers a new query for each category
    items = await category.items.all()
    for item in items:
        print(f"  Item: {item.name}")
```

✅ GOOD: Eager loading with `select_related` or `fetch_related`
```python
# Using select_related for forward relations (e.g., Item -> Category)
items_with_categories = await Item.all().select_related('category')
for item in items_with_categories:
    print(f"Item: {item.name}, Category: {item.category.name}") # category is pre-fetched

# Using fetch_related for reverse relations (e.g., Category -> items)
categories_with_items = await Category.all()
await categories_with_items.fetch_related('items') # Fetches all items for all categories in one query
for category in categories_with_items:
    print(f"Category: {category.name}")
    for item in category.items:
        print(f"  Item: {item.name}")
```

## 5. Batch Operations

Use `bulk_create` and `bulk_update` for inserting/updating multiple records efficiently.

❌ BAD: Individual saves in a loop
```python
# Inefficient for many objects
for data in list_of_data:
    await MyModel.create(**data)
```

✅ GOOD: Batch operations
```python
from datetime import datetime

# Efficient for many objects
objects_to_create = [MyModel(**data) for data in list_of_data]
await MyModel.bulk_create(objects_to_create)

# Similarly for updates:
objects_to_update = await MyModel.filter(status="pending")
for obj in objects_to_update:
    obj.status = "processed"
    obj.updated_at = datetime.now()
await MyModel.bulk_update(objects_to_update, fields=["status", "updated_at"])
```

## 6. Transactions for Atomicity

Ensure data consistency with atomic transactions for related operations.

```python
from tortoise.transactions import in_transaction

class Account(Model): # Assume Account model exists
    id: int = fields.IntField(primary_key=True)
    balance: float = fields.DecimalField(max_digits=10, decimal_places=2)

async def transfer_funds(from_account_id: int, to_account_id: int, amount: float):
    async with in_transaction() as connection:
        from_account = await Account.get(id=from_account_id)
        to_account = await Account.get(id=to_account_id)

        if from_account.balance < amount:
            raise ValueError("Insufficient funds")

        from_account.balance -= amount
        to_account.balance += amount

        await from_account.save(update_fields=["balance"], using_db=connection)
        await to_account.save(update_fields=["balance"], using_db=connection)
    # Transaction commits automatically on success, rolls back on error
```

## 7. Version Pinning

Pin your `tortoise-orm` version in `requirements.txt` to a stable minor release (e.g., `0.25.x`) to avoid unexpected breaking changes.

❌ BAD: Loose dependency
```
tortoise-orm
```

✅ GOOD: Pinned dependency
```
tortoise-orm==0.25.10 # Pin to a specific patch version
# or
tortoise-orm~=0.25.0 # Pin to a minor version range
```