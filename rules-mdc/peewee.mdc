---
description: Definitive guidelines for writing robust, maintainable, and performant Peewee ORM code in Python, covering data modeling, querying, and application structure.
globs: **/*.py
---
# peewee Best Practices

Peewee is a lightweight, expressive ORM for Python. These guidelines ensure your Peewee code is clean, efficient, and aligns with modern best practices as of 2025.

---

## 1. Code Organization and Structure

Always establish a clear, DRY (Don't Repeat Yourself) structure for your Peewee models and database connection.

### 1.1 Define a `BaseModel` for Database Connection

Centralize your database instance within a `BaseModel` to avoid repetitive declarations and ensure all models use the same connection.

❌ BAD: Direct database assignment in every model
```python
from peewee import *

db = SqliteDatabase('app.db')

class User(Model):
    username = CharField()
    class Meta:
        database = db # Repeated for every model
```

✅ GOOD: Inherit from a `BaseModel`
```python
from peewee import *

db = SqliteDatabase('app.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    username = CharField(unique=True)
```

### 1.2 Model Naming Convention

Name your model classes in the singular form. Peewee automatically pluralizes table names (e.g., `User` -> `user`).

❌ BAD: Plural model name
```python
class Users(BaseModel): # Table might be 'userss' or 'users' depending on backend
    name = CharField()
```

✅ GOOD: Singular model name
```python
class User(BaseModel): # Table will be 'user'
    name = CharField()
```

### 1.3 Explicit Imports for Model Definitions

For model definition files, use `from peewee import *` for convenience. This is a common and accepted pattern within the Peewee ecosystem. For other modules or specific utility functions, prefer explicit imports.

❌ BAD: Overly verbose imports for model definitions
```python
from peewee import Model, CharField, IntegerField, SqliteDatabase, DateTimeField # ... and so on
```

✅ GOOD: Concise model definition imports
```python
from peewee import * # For model definition files (e.g., models.py)
import datetime

# In other files (e.g., main.py, services.py), for specific functions/classes:
from peewee import SqliteDatabase, DoesNotExist
```

## 2. Data Modeling

Define robust models with clear field types, constraints, and relationships.

### 2.1 Explicit Field Constraints and Defaults

Always define fields with appropriate Python types, `unique=True` where necessary, and sensible `default` values. Peewee uses `id` as an auto-incrementing primary key by default if none is specified.

```python
import datetime
from peewee import *

db = SqliteDatabase('app.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    username = CharField(unique=True, index=True) # Ensure uniqueness and fast lookups
    email = CharField(unique=True, null=False)
    created_at = DateTimeField(default=datetime.datetime.now) # Use datetime.datetime.now, not datetime.now()
    is_active = BooleanField(default=True)

class Post(BaseModel):
    user = ForeignKeyField(User, backref='posts', on_delete='CASCADE') # Define relationship and cascade delete
    title = CharField(max_length=255, null=False)
    content = TextField(null=False)
    published_at = DateTimeField(null=True) # Can be null until published
    views = IntegerField(default=0)
```

### 2.2 Foreign Key Relationships with `backref` and `on_delete`

Always define `ForeignKeyField` with a `backref` to easily access related objects from the parent model. Specify `on_delete` for database-level referential integrity.

❌ BAD: Missing `backref` and `on_delete`
```python
class Comment(BaseModel):
    post = ForeignKeyField(Post) # Cannot easily get comments from a Post instance
    text = TextField()
```

✅ GOOD: With `backref` and `on_delete`
```python
class Comment(BaseModel):
    post = ForeignKeyField(Post, backref='comments', on_delete='CASCADE') # `post.comments` will work
    author = ForeignKeyField(User, backref='comments', on_delete='SET NULL', null=True) # Example with SET NULL
    text = TextField()
```

## 3. Database Connection and Transactions

Manage your database connections and ensure data integrity with transactions.

### 3.1 Connect and Create Tables Once at Application Startup

Establish the database connection and create tables only once when your application initializes. Close the connection if not using persistent connections or a framework that manages it.

❌ BAD: Connecting/creating tables in every script or function
```python
# In script A
db.connect()
db.create_tables([User, Post])

# In script B
db.connect() # Redundant, potential for errors
```

✅ GOOD: Centralized application startup
```python
# app_init.py (or main entry point)
from peewee import *
import datetime

db = SqliteDatabase('app.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    username = CharField(unique=True)

class Post(BaseModel):
    user = ForeignKeyField(User, backref='posts')
    title = CharField()

def initialize_db():
    db.connect()
    db.create_tables([User, Post]) # Pass all models here
    db.close() # Close immediately if not using persistent connections

if __name__ == '__main__':
    initialize_db()
    print("Database initialized.")
    # Now, in your application logic, you'd open/close connections per request/operation
    # or use a connection pool.
```

### 3.2 Use Transactions for Atomic Operations

Wrap any multi-step database modifications in a transaction (`with db.atomic():`) to guarantee atomicity and improve performance.

❌ BAD: Multiple, non-atomic operations
```python
user = User.create(username='alice')
Post.create(user=user, title='First post') # If this fails, user is still created, leading to inconsistency
```

✅ GOOD: Atomic transaction
```python
try:
    with db.atomic():
        user = User.create(username='bob')
        Post.create(user=user, title='Second post') # Both succeed or both fail
except IntegrityError:
    print("Failed to create user and post due to integrity error.")
```

## 4. Query Optimization

Write efficient queries to minimize database load and improve response times.

### 4.1 Avoid N+1 Query Problem

Use `join()` or `prefetch()` to load related objects in a single query, preventing the N+1 problem.

❌ BAD: N+1 queries
```python
# Fetching posts, then iterating to get each user's username
for post in Post.select():
    print(f"{post.title} by {post.user.username}") # Each post.user.username triggers a new query
```

✅ GOOD: Single query with `join()`
```python
for post in Post.select().join(User):
    print(f"{post.title} by {post.user.username}") # User data loaded with post
```

### 4.2 Use Bulk Operations for Inserts

For inserting many records, use `bulk_create()` for significant performance gains by reducing the number of database round trips.

❌ BAD: Individual `create()` calls in a loop
```python
users_data = [{'username': f'user_{i}'} for i in range(100)]
for data in users_data:
    User.create(**data) # 100 separate INSERT statements
```

✅ GOOD: Single `bulk_create()` call
```python
users_data = [{'username': f'user_{i}'} for i in range(100)]
User.bulk_create([User(**data) for data in users_data]) # Single INSERT statement (or few, depending on batch size)
```

### 4.3 Atomic Updates

Perform updates atomically using `Model.update().where().execute()` to prevent race conditions and improve efficiency.

❌ BAD: Fetch-modify-save (prone to race conditions)
```python
post = Post.get(Post.id == 1)
post.views += 1
post.save() # Another process could have incremented views between get and save
```

✅ GOOD: Atomic update
```python
Post.update(views=Post.views + 1).where(Post.id == 1).execute() # Single, atomic operation
```

### 4.4 Pagination

Use the built-in `paginate()` method for efficient result set slicing, especially for large datasets.

```python
# Get page 2, with 20 items per page, ordered by username
users_page_2 = User.select().order_by(User.username).paginate(2, 20)
```

## 5. Security Best Practices

Peewee handles SQL injection prevention by default through parameterized queries. Focus on proper input validation in your application logic.

### 5.1 Trust Peewee's Query Builder

Avoid constructing SQL strings manually. Always use Peewee's ORM methods and expressions, which automatically parameterize inputs.

❌ BAD: Manual string formatting (vulnerable to injection)
```python
# DO NOT DO THIS. This is a severe security vulnerability.
user_input = "'; DROP TABLE users; --"
query = f"SELECT * FROM user WHERE username = '{user_input