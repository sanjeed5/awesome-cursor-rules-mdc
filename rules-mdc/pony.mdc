---
description: Definitive guidelines for writing efficient, maintainable, and secure Python applications using Pony ORM.
globs: **/*.py
---
# pony Best Practices

Pony ORM simplifies database interactions with Pythonic syntax and powerful optimizations. Adhere to these rules for robust, high-performance applications.

## 1. Always Use `db_session` for Transaction Management

The `db_session` context manager is non-negotiable. It ensures proper transaction handling, automatic commits, and rollbacks, preventing resource leaks and data inconsistencies.

❌ BAD: Direct database operations without a session.
```python
from pony.orm import *
from decimal import Decimal

db = Database('sqlite', ':memory:')

class Product(db.Entity):
    name = Required(str)
    price = Required(Decimal)

db.generate_mapping(create_tables=True)

# This will fail or lead to unexpected behavior in production
p = Product(name='Laptop', price=Decimal('1200.00'))
db.commit() # Manual commit is error-prone and easily forgotten
```

✅ GOOD: Encapsulate all database work within `db_session`.
```python
from pony.orm import *
from decimal import Decimal

db = Database('sqlite', ':memory:')

class Product(db.Entity):
    name = Required(str)
    price = Required(Decimal)

db.generate_mapping(create_tables=True)

@db_session
def create_product(name: str, price: Decimal) -> Product:
    p = Product(name=name, price=price)
    return p # Auto-committed on exit if no exception

with db_session:
    laptop = create_product('Laptop', Decimal('1200.00'))
    print(f"Created product: {laptop.name}")
```

## 2. Leverage Pythonic Query Syntax for Optimization

Pony translates generator expressions and lambdas into optimized SQL, automatically handling N+1 problems and complex joins. Avoid manual SQL unless absolutely necessary.

❌ BAD: Manual filtering or inefficient data access.
```python
@db_session
def get_expensive_products_bad():
    all_products = list(select(p for p in Product)) # Fetches all products
    expensive_products = [p for p in all_products if p.price > Decimal('1000')] # Filters in Python
    return expensive_products
```

✅ GOOD: Let Pony generate optimized SQL.
```python
@db_session
def get_expensive_products_good() -> list[Product]:
    # Pony translates this directly to SQL: SELECT ... WHERE price > 1000
    return select(p for p in Product if p.price > Decimal('1000'))[:] # Use [:] to fetch all results
```

## 3. Separate Data Access from Business Logic

Maintain a clean architecture by defining entities in a dedicated `models.py` and encapsulating data operations in a `repositories.py` or `services.py` module.

❌ BAD: Mixing entity definitions, database setup, and business logic in one file.
```python
# app.py
from pony.orm import *
db = Database('sqlite', 'app.sqlite')
class User(db.Entity):
    name = Required(str)
    email = Required(str, unique=True)
db.generate_mapping(create_tables=True)

@db_session
def create_user_and_send_email(name, email):
    user = User(name=name, email=email)
    # ... send email logic here ...
```

✅ GOOD: Clear separation of concerns.
```python
# models.py
from pony.orm import *
db = Database() # Bind later

class User(db.Entity):
    id = PrimaryKey(int, auto=True)
    name = Required(str)
    email = Required(str, unique=True)

# database.py
from pony.orm import Database
from .models import db # Import db instance

def setup_database(provider: str, filename: str):
    db.bind(provider=provider, filename=filename, create_db=True)
    db.generate_mapping(create_tables=True)

# services.py
from pony.orm import db_session
from .models import User
from typing import List

@db_session
def create_user(name: str, email: str) -> User:
    user = User(name=name, email=email)
    return user

@db_session
def get_all_users() -> List[User]:
    return list(select(u for u in User))

# main.py (or app.py)
from .database import setup_database
from .services import create_user, get_all_users

if __name__ == '__main__':
    setup_database('sqlite', 'my_app.sqlite')
    
    create_user('Alice', 'alice@example.com')
    create_user('Bob', 'bob@example.com')
    
    for user in get_all_users():
        print(f"User: {user.name}, Email: {user.email}")
```

## 4. Apply Static Typing to Entities and Queries

Use type hints (`typing` module) for entity attributes and function signatures. This improves readability, enables IDE auto-completion, and catches errors early.

❌ BAD: Untyped entity attributes and function parameters.
```python
class Order(db.Entity):
    amount = Required(float) # float is imprecise for currency
    customer = Required('Customer')

@db_session
def get_total_orders(customer_id): # No type hints
    customer = Customer[customer_id]
    return sum(o.amount for o in customer.orders)
```

✅ GOOD: Explicit type hints for clarity and correctness.
```python
from decimal import Decimal
from typing import List

class Customer(db.Entity):
    id = PrimaryKey(int, auto=True)
    name = Required(str)
    orders: List['Order'] # Forward reference for type hinting

class Order(db.Entity):
    id = PrimaryKey(int, auto=True)
    amount = Required(Decimal) # Use Decimal for currency
    customer = Required(Customer)

@db_session
def get_customer_total_orders(customer_id: int) -> Decimal:
    customer: Customer = Customer[customer_id]
    # Pony handles aggregation efficiently
    total_amount: Decimal = sum(o.amount for o in customer.orders)
    return total_amount
```

## 5. Utilize Pony's Automatic Caching and IdentityMap

Pony automatically caches entities and query results within a `db_session`. Avoid manual caching layers for simple entity lookups to prevent stale data.

❌ BAD: Implementing a custom cache for entities within a session.
```python
my_entity_cache = {} # Don't do this; it's redundant and error-prone with Pony

@db_session
def get_user_cached(user_id: int) -> User:
    if user_id not in my_entity_cache:
        user = User[user_id]
        my_entity_cache[user_id] = user
    return my_entity_cache[user_id]
```

✅ GOOD: Trust Pony's IdentityMap and session-level caching.
```python
@db_session
def get_user_pony_cached(user_id: int) -> User:
    # Pony's IdentityMap ensures that User[user_id] returns the same object
    # within the session if it has already been fetched.
    user1 = User[user_id]
    user2 = User[user_id]
    assert user1 is user2 # This assertion is True within the same db_session
    return user1
```

## 6. Manage Schema with the Online Editor or Versioned Scripts

Keep your Pony model definitions in sync with the database schema. For complex projects, use version-controlled migration scripts. For rapid prototyping, Pony's online editor is excellent.

❌ BAD: Manually altering database tables without updating Pony models, or vice-versa.
```python
# models.py
class Product(db.Entity):
    name = Required(str)
    # Forgot to add 'description' here, but added it to DB manually, leading to mismatches.
```

✅ GOOD: Use `db.generate_mapping(create_tables=True)` for initial setup and Pony's editor or custom migration scripts for changes.
```python
# Initial setup
db.generate_mapping(create_tables=True)

# For schema changes, either use Pony's online editor to generate SQL
# or write explicit migration scripts (e.g., using Alembic or custom logic)
# to apply changes and then update your Pony entity definitions.
```