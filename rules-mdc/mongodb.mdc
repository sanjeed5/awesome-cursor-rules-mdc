---
description: This guide provides definitive best practices for designing, querying, and maintaining MongoDB databases to ensure high performance, scalability, and maintainability in cloud-native applications.
globs: **/*
---
# mongodb Best Practices

MongoDB's flexible schema is a superpower, but with great power comes great responsibility. This guide cuts through the noise to give you the definitive, opinionated best practices for building robust, high-performance applications with MongoDB. Follow these rules to avoid common pitfalls and leverage MongoDB effectively.

## 1. Data Modeling: Store Data Accessed Together, Together

The cardinal rule: embed data that is accessed together within a single document. This minimizes joins and maximizes read performance. Use referencing for many-to-many relationships or when embedded data would exceed the 16MB document limit or grow unboundedly.

### 1.1. Embed for Read Performance

✅ **GOOD**: Embed tightly coupled, frequently accessed data.
```json
// User document with embedded preferences
{
    "_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b"),
    "username": "johndoe",
    "email": "john.doe@example.com",
    "preferences": {
        "theme": "dark",
        "notifications": {
            "email": true,
            "sms": false
        }
    }
}
```
❌ **BAD**: Separating tightly coupled data into multiple collections, forcing `$lookup` for common reads.
```json
// User document
{ "_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b"), "username": "johndoe", "email": "john.doe@example.com" }
// UserPreferences document (separate collection)
{ "_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5c"), "userId": ObjectId("656e2c0e8a7b9c1d2e3f4a5b"), "theme": "dark", "notifications": { "email": true, "sms": false } }
// Requires a join to get user with preferences.
```

### 1.2. Reference for Unbounded Growth or Many-to-Many

Use referencing when a sub-document array could grow indefinitely (e.g., comments on a blog post) or when documents are frequently updated independently.

✅ **GOOD**: Reference comments from a blog post.
```json
// Post document
{
    "_id": ObjectId("post123"),
    "title": "My Awesome Post",
    "content": "...",
    "authorId": ObjectId("user456"),
    "commentCount": 5 // Denormalized for quick access
}
// Comment document (separate collection)
{
    "_id": ObjectId("comment789"),
    "postId": ObjectId("post123"),
    "authorId": ObjectId("user987"),
    "text": "Great post!",
    "createdAt": ISODate("2025-01-01T10:00:00Z")
}
```
❌ **BAD**: Embedding an unbounded array of comments within the post document. This risks exceeding the 16MB document limit and makes updates inefficient.
```json
// Post document with embedded comments
{
    "_id": ObjectId("post123"),
    "title": "My Awesome Post",
    "content": "...",
    "authorId": ObjectId("user456"),
    "comments": [ // This array can grow indefinitely!
        {"authorId": ObjectId("user987"), "text": "Great post!", "createdAt": ISODate("2025-01-01T10:00:00Z")},
        {"authorId": ObjectId("user111"), "text": "I agree!", "createdAt": ISODate("2025-01-01T10:05:00Z")}
    ]
}
```

### 1.3. Enforce Schema with Validation

While MongoDB is schema-less, enforce expected field types and presence using schema validation rules at the collection level. This improves data consistency and prevents application errors.

✅ **GOOD**: Use schema validation to ensure `username` is a string and `email` is required.
```javascript
// In MongoDB shell or Compass
db.createCollection("users", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["username", "email"],
            properties: {
                username: {
                    bsonType: "string",
                    description: "must be a string and is required"
                },
                email: {
                    bsonType: "string",
                    pattern: "^.+@.+\\..+$",
                    description: "must be a valid email address and is required"
                }
            }
        }
    }
})
```
❌ **BAD**: Relying solely on application-level validation, leading to inconsistent data in the database.

## 2. Query Optimization: Fetch Only What You Need

Efficient queries are critical. Always project fields, avoid slow operators, and leverage the aggregation pipeline.

### 2.1. Use Projections to Limit Returned Fields

Always specify which fields to return. Retrieving entire documents when only a few fields are needed wastes network bandwidth and memory.

✅ **GOOD**: Fetch only `username` and `email`.
```python
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient("mongodb://localhost:27017/")
users_collection = client.mydatabase.users

user_data = users_collection.find_one(
    {"_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b")},
    {"username": 1, "email": 1, "_id": 0} # Exclude _id explicitly if not needed
)
```
❌ **BAD**: Fetching the entire document unnecessarily.
```python
user_data = users_collection.find_one({"_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b")}) # Returns all fields
```

### 2.2. Avoid `$where` and JavaScript in Queries

`$where` requires MongoDB to execute JavaScript for each document, preventing index usage and severely impacting performance. Use native query operators instead.

✅ **GOOD**: Use native operators for filtering.
```python
from datetime import datetime, timedelta

users_collection.find({"createdAt": {"$lt": datetime.now() - timedelta(days=30)}})
```
❌ **BAD**: Using `$where` for date comparison.
```python
users_collection.find({"$where": "this.createdAt < new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)"})
```

### 2.3. Leverage the Aggregation Pipeline

For complex data transformations, reporting, or analytics, use the aggregation pipeline. It processes data efficiently on the server, minimizing data transfer.

✅ **GOOD**: Aggregate user activity.
```python
pipeline = [
    {"$match": {"status": "active"}},
    {"$group": {"_id": "$country", "totalUsers": {"$sum": 1}}},
    {"$sort": {"totalUsers": -1}}
]
result = list(users_collection.aggregate(pipeline))
```
❌ **BAD**: Fetching large datasets and performing complex aggregations in application code.

## 3. Indexing: Speed Up Your Reads

Indexes are your best friend for query performance. Use them strategically.

### 3.1. Index Frequently Queried Fields

Create indexes on fields used in `find()` queries, `sort()` operations, and aggregation pipeline stages like `$match` or `$sort`.

✅ **GOOD**: Index `email` for fast lookups.
```python
from pymongo import ASCENDING

users_collection.create_index([("email", ASCENDING)], unique=True)
```
❌ **BAD**: No index on a frequently queried field, leading to collection scans.

### 3.2. Prefer Compound Indexes for Multi-Field Queries

If you frequently query on multiple fields, a compound index can cover multiple query patterns efficiently. Order fields in the compound index by cardinality and query selectivity.

✅ **GOOD**: Compound index for `status` and `createdAt`.
```python
from pymongo import DESCENDING

posts_collection = client.mydatabase.posts
posts_collection.create_index([("status", ASCENDING), ("createdAt", DESCENDING)])
# This index supports queries like:
# find({"status": "published", "createdAt": {"$gt": some_date}})
# find({"status": "published"})
# find({}, sort=[("status", 1), ("createdAt", -1)])
```
❌ **BAD**: Separate single-field indexes for `status` and `createdAt` when they are often queried together.

### 3.3. Use `explain()` to Validate Index Usage

Always use `explain()` to understand how MongoDB executes your queries and confirm that indexes are being used effectively.

✅ **GOOD**: Analyze query plan.
```python
query_plan = users_collection.find({"email": "john.doe@example.com"}).explain()
# Look for "IXSCAN" in the winning plan's "stage" field.
```
❌ **BAD**: Assuming indexes are working without verification, leading to hidden performance bottlenecks.

## 4. Code Organization: Use ODMs, Fallback to Driver

For most CRUD operations, use an Object Document Mapper (ODM) for schema validation and cleaner code. For complex or highly optimized operations, use the native driver.

### 4.1. Use an ODM for Schema-Driven Development (e.g., MongoEngine for Python)

ODMs provide a layer of abstraction, define schemas, and handle common operations, making your code more maintainable and less error-prone.

✅ **GOOD**: Define models with MongoEngine.
```python
from mongoengine import Document, StringField, EmailField, DateTimeField
from datetime import datetime

class User(Document):
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    created_at = DateTimeField(default=datetime.now)

# Usage (after connecting MongoEngine):
# user = User(username="janedoe", email="jane.doe@example.com")
# user.save()
```
❌ **BAD**: Manually managing document structure and validation with raw PyMongo for every operation.

### 4.2. Fallback to Native Driver for Fine-Grained Control

When an ODM's abstraction limits performance or expressiveness for a specific complex query (e.g., advanced aggregation pipelines or bulk writes), use the native driver directly.

✅ **GOOD**: Use PyMongo for a complex aggregation.
```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.mydatabase
users_collection = db.users

pipeline = [
    {"$match": {"status": "active"}},
    {"$group": {"_id": "$country", "totalUsers": {"$sum": 1}}}
]
result = list(users_collection.aggregate(pipeline))
```
❌ **BAD**: Forcing complex operations into an ODM's ORM-like syntax, potentially leading to inefficient queries or verbose code.

## 5. Security Best Practices: Lock It Down

Security is paramount. Implement role-based access control, enable TLS, and manage read/write concerns.

### 5.1. Implement Role-Based Access Control (RBAC)

Grant users and applications the minimum necessary privileges. Never use root or administrative credentials for application-level access.

✅ **GOOD**: Create specific roles.
```javascript
// In MongoDB shell
db.createUser(
   {
     user: "appUser",
     pwd: passwordPrompt(), // Use a strong password
     roles: [ { role: "readWrite", db: "mydatabase" } ]
   }
)
```
❌ **BAD**: Using a single user with `dbAdmin` or `readWriteAnyDatabase` for all application interactions.

### 5.2. Enable TLS/SSL for All Connections

Encrypt all data in transit between your application and MongoDB.

✅ **GOOD**: Connect with TLS enabled.
```python
client = MongoClient("mongodb://localhost:27017/", tls=True, tlsCAFile="path/to/ca.pem")
```
❌ **BAD**: Connecting without TLS, exposing sensitive data to eavesdropping.

## 6. Common Pitfalls & Gotchas: Avoid These Traps

### 6.1. Avoid Multi-Document Transactions Unless Absolutely Necessary

Single-document operations in MongoDB are atomic. Multi-document transactions add overhead and complexity. Only use them when atomicity across multiple documents or collections is a strict requirement.

✅ **GOOD**: Rely on single-document atomicity for most updates.
```python
users_collection.update_one({"_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b")}, {"$set": {"status": "inactive"}}) # Atomic
```
❌ **BAD**: Wrapping simple, single-document updates in a multi-document transaction for no real benefit.

### 6.2. Manage Cursor Lifecycles

Ensure cursors are closed or fully iterated, especially in long-running processes, to prevent resource leaks on the server.

✅ **GOOD**: Iterate fully or close explicitly.
```python
cursor = users_collection.find({"status": "active"})
for user in cursor:
    process_user(user)
cursor.close() # Explicitly close if not fully iterated
```
❌ **BAD**: Leaving cursors open indefinitely, consuming server resources.

## 7. Testing Approaches: Isolate Your Database

Focus on integration tests for database interactions and mock the database for unit tests.

### 7.1. Use an In-Memory Mock for Unit Tests

For unit testing business logic that interacts with MongoDB, mock the database client or ODM to ensure tests are fast and isolated.

✅ **GOOD**: Mock `find_one` for a unit test.
```python
from unittest.mock import MagicMock, patch
from bson.objectid import ObjectId

# Assume get_user_by_id is a function in my_app.py that uses users_collection
# from pymongo.collection import Collection
# users_collection: Collection = ... # Initialized elsewhere

def get_user_by_id(user_id: ObjectId):
    return users_collection.find_one({"_id": user_id})

def test_get_user_by_id_mocked():
    mock_user_data = {"_id": ObjectId("656e2c0e8a7b9c1d2e3f4a5b"), "username": "testuser"}
    mock_collection = MagicMock()
    mock_collection.find_one.return_value = mock_user_data

    with patch('__main__.users_collection', mock_collection): # Adjust patch path as needed
        user = get_user_by_id(ObjectId("656e2c0e8a7b9c1d2e3f4a5b"))
        assert user["username"] == "testuser"
        mock_collection.find_one.assert_called_once()
```
❌ **BAD**: Connecting to a real (even local) MongoDB instance for every unit test, making tests slow and brittle.

By adhering to these best practices, you'll build performant, scalable, and maintainable applications with MongoDB.