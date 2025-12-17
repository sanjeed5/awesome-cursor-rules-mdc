---
description: This guide enforces modern SQLAlchemy 2.x best practices for Python applications, ensuring type-safe, performant, and maintainable database interactions.
globs: **/*.py
---
# SQLAlchemy Best Practices (2.x Style)

This document outlines the definitive best practices for using SQLAlchemy 2.x in our projects. We exclusively adopt the "2-style" API, leveraging its fully typed, declarative features for robust and maintainable code.

## 1. Code Organization and Data Modeling

Always define your ORM models using the modern `DeclarativeBase` and `Mapped` annotations. Keep model definitions in a dedicated `models.py` or `orm_models/` directory.

### 1.1 Declarative Models with Type Annotations

Use `DeclarativeBase` as your base class and `Mapped` for all ORM-mapped attributes. This enables static type checking and clear schema definition.

❌ BAD: Legacy `declarative_base()` function and untyped `Column`.
```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String) # No type hint
```

✅ GOOD: Modern `DeclarativeBase` and `Mapped` for explicit typing.
```python
from typing import List, Optional
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, func, DateTime

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    created_at: Mapped[DateTime] = mapped_column(DateTime, insert_default=func.now())

    addresses: Mapped[List["Address"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))

    user: Mapped["User"] = relationship(back_populates="addresses")
```

### 1.2 Mixins for Common Fields

Implement common fields like timestamps, soft-delete flags, or version counters using mixins.

```python
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, func

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now(), onupdate=func.now())

class SoftDeleteMixin:
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)

class User(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    # ... other fields
```

## 2. Session Management

The `Session` is your unit of work. Always use it as a context manager to ensure proper transaction handling and resource cleanup.

### 2.1 Context Manager for Sessions

Wrap all database operations within a `with Session(engine) as session:` block. This guarantees `commit()` on success and `rollback()` on error, followed by `close()`.

❌ BAD: Manual session handling, prone to leaks and uncommitted transactions.
```python
session = Session(engine)
try:
    user = User(name="Alice")
    session.add(user)
    session.commit()
except Exception:
    session.rollback()
finally:
    session.close()
```

✅ GOOD: `Session` as a context manager.
```python
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine("sqlite:///./test.db")
Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_user(name: str):
    with Session() as session:
        user = User(name=name)
        session.add(user)
        session.commit() # Flushes changes and commits transaction
        session.refresh(user) # Refresh to get auto-generated IDs
        return user
```

### 2.2 Explicit Transaction Blocks

For complex operations requiring multiple steps within a single transaction, use `session.begin()`.

```python
def transfer_funds(from_account_id: int, to_account_id: int, amount: float):
    with Session() as session:
        with session.begin(): # Explicit transaction block
            from_account = session.get(Account, from_account_id)
            to_account = session.get(Account, to_account_id)

            if not from_account or not to_account:
                raise ValueError("Account not found")
            if from_account.balance < amount:
                raise ValueError("Insufficient funds")

            from_account.balance -= amount
            to_account.balance += amount
        # Transaction commits here if no exceptions, or rolls back if an error occurs
    print(f"Transferred {amount} from {from_account_id} to {to_account_id}")
```

## 3. Querying and Optimization

Always use the `select()` construct for queries and prioritize eager loading for relationships to avoid N+1 problems.

### 3.1 Use `select()` for All Queries

The `Query` object is deprecated in 2.x. Use `select()` for all ORM and Core queries.

❌ BAD: Using the legacy `session.query()` API.
```python
users = session.query(User).filter(User.name == "Alice").all()
```

✅ GOOD: Using `select()` with `session.execute()`.
```python
from sqlalchemy import select

def get_user_by_name(session: Session, name: str) -> Optional[User]:
    stmt = select(User).where(User.name == name)
    return session.scalar(stmt) # Use scalar for single result, one_or_none for one, all for list
```

### 3.2 Eager Loading Relationships

Prevent N+1 query issues by eagerly loading related objects using `selectinload` or `joinedload`. `selectinload` is generally preferred for collections.

❌ BAD: Lazy loading in a loop, leading to N+1 queries.
```python
users = session.scalars(select(User)).all()
for user in users:
    print(f"{user.name} has {len(user.addresses)} addresses") # Each access triggers a new query
```

✅ GOOD: Eager loading with `selectinload`.
```python
from sqlalchemy.orm import selectinload

def get_users_with_addresses(session: Session) -> List[User]:
    stmt = select(User).options(selectinload(User.addresses))
    return session.scalars(stmt).all()

# Now, accessing user.addresses won't trigger additional queries
users = get_users_with_addresses(session)
for user in users:
    print(f"{user.name} has {len(user.addresses)} addresses")
```

## 4. Data Manipulation

The ORM Unit of Work pattern handles inserts, updates, and deletes efficiently.

### 4.1 Adding and Updating Objects

Add new objects with `session.add()`. Modifications to existing objects are tracked automatically.

```python
def update_user_name(session: Session, user_id: int, new_name: str):
    user = session.get(User, user_id) # Efficiently get by primary key
    if user:
        user.name = new_name # Change is tracked
        session.commit()
    return user

def add_new_user_with_address(session: Session, name: str, email: str):
    user = User(name=name, fullname=name)
    address = Address(email_address=email, user=user) # Relationship automatically links
    session.add(user) # Adding user also adds address due to cascade
    session.commit()
    return user
```

### 4.2 Deleting Objects

Delete objects using `session.delete()`.

```python
def delete_user(session: Session, user_id: int):
    user = session.get(User, user_id)
    if user:
        session.delete(user)
        session.commit()
        print(f"User {user_id} deleted.")
    else:
        print(f"User {user_id} not found.")
```

### 4.3 Bulk Operations (Use Sparingly)

Only use bulk operations (`session.bulk_insert_mappings`, `session.execute(insert/update/delete)`) when performance is paramount and ORM event handling (e.g., `onupdate` hooks) is not required.

```python
from sqlalchemy import insert

def bulk_insert_users(session: Session, user_data: List[dict]):
    # This bypasses ORM object creation and event hooks.
    # Use when you have raw data and need speed.
    session.execute(insert(User), user_data)
    session.commit()

# Example user_data: [{"name": "Charlie", "fullname": "Charlie Brown"}, ...]
```

## 5. Migrations and Type Checking

Integrate Alembic for schema migrations and ensure static type checking with Mypy or Ruff.

### 5.1 Alembic for Schema Evolution

Always use Alembic to manage database schema changes. Never manually alter the database schema.

```bash
# Initialize Alembic (once per project)
alembic init -t async migrations

# Generate a new migration script
alembic revision --autogenerate -m "Add new_column to user_account"

# Apply migrations
alembic upgrade head
```

### 5.2 Static Type Checking

Leverage Python's `typing` and SQLAlchemy's `Mapped` for comprehensive type checking with tools like Mypy or Ruff. This catches schema mismatches and API misuses early.

```python
# mypy.ini or pyproject.toml configuration for mypy
[mypy]
plugins = sqlalchemy.ext.mypy.plugin
```