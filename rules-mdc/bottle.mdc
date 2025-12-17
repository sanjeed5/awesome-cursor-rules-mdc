---
description: Definitive guidelines for building maintainable, secure, and performant Bottle applications in 2025, emphasizing modern Python best practices.
globs: **/*.py
---
# bottle Best Practices

Bottle is a robust micro-framework for Python backends. To leverage it effectively in 2025, adhere to these opinionated guidelines, focusing on structure, maintainability, and security.

## 1. Code Organization and Structure

Organize your Bottle project as a Python package with a clear separation of concerns. Avoid monolithic `app.py` files.

**✅ GOOD: Modular Project Structure**

```
my_bottle_app/
├── app.py              # Application entry point, config loading, plugin setup
├── routes/             # API endpoint definitions
│   ├── __init__.py
│   └── users.py
│   └── products.py
├── services/           # Business logic, database interactions
│   ├── __init__.py
│   └── user_service.py
│   └── product_service.py
├── schemas/            # Pydantic models for request/response validation
│   ├── __init__.py
│   └── user_schema.py
├── config/             # Configuration files (e.g., default.ini)
│   └── default.ini
├── tests/              # Unit and integration tests
│   └── test_users.py
└── requirements.txt
```

**❌ BAD: Monolithic `app.py`**

```python
# app.py (❌ BAD)
from bottle import route, run, request, Bottle

app = Bottle()

# All routes, logic, and config in one file
@app.route('/users', method='GET')
def get_users():
    # ... database logic, business logic, everything here ...
    return {'users': []}

# ... many more routes and intertwined logic ...

if __name__ == '__main__':
    run(app, host='localhost', port=8080)
```

**✅ GOOD: Structured `app.py` and `routes/users.py`**

```python
# my_bottle_app/app.py (✅ GOOD)
import os
from bottle import Bottle, run
from my_bottle_app.routes import users
from my_bottle_app.plugins.db import DatabasePlugin # Example plugin

app = Bottle()

# 1. Load default config
app.config.load_config('my_bottle_app/config/default.ini')

# 2. Override with environment variables
app.config['database.url'] = os.environ.get('DATABASE_URL', app.config.get('database.url'))
app.config['debug'] = bool(os.environ.get('DEBUG', app.config.get('debug', False)))

# 3. Install plugins
app.install(DatabasePlugin(app.config['database.url']))

# 4. Mount route modules
app.mount('/users', users.app) # users.app is a Bottle instance from routes/users.py

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=8080, debug=app.config['debug'])
```

```python
# my_bottle_app/routes/users.py (✅ GOOD)
from bottle import Bottle, request, response, HTTPResponse
from typing import List
from pydantic import ValidationError
from my_bottle_app.schemas.user_schema import UserCreate, UserResponse
from my_bottle_app.services.user_service import UserService

app = Bottle()

@app.get('/')
def get_all_users() -> List[UserResponse]:
    """Retrieves all users."""
    user_service = UserService(request.db) # Dependency injection via plugin
    users = user_service.get_all()
    return [UserResponse.model_validate(u).model_dump() for u in users]

@app.post('/')
def create_user() -> UserResponse:
    """Creates a new user."""
    try:
        user_data = UserCreate.model_validate(request.json)
    except ValidationError as e:
        raise HTTPResponse(status=400, body={'error': e.errors()})

    user_service = UserService(request.db)
    new_user = user_service.create(user_data)
    response.status = 201
    return UserResponse.model_validate(new_user).model_dump()
```

## 2. Configuration Management

Centralize all application settings in `app.config`. Load defaults from `.ini` files, then override with environment variables. Keep `app.config` immutable after initial load.

**❌ BAD: Hardcoded values and mutable globals**

```python
# ❌ BAD
DB_URL = "sqlite:///./test.db" # Hardcoded
DEBUG_MODE = True

def get_db_connection():
    # ... uses DB_URL ...
    pass
```

**✅ GOOD: `app.config` with environment overrides**

```python
# my_bottle_app/config/default.ini
[app]
debug = False
log_level = INFO

[database]
url = sqlite:///./default.db

# my_bottle_app/app.py (snippet)
import os
from bottle import Bottle

app = Bottle()
app.config.load_config('my_bottle_app/config/default.ini')

# Override with environment variables
app.config['app.debug'] = os.environ.get('DEBUG', app.config['app.debug']) == 'True'
app.config['database.url'] = os.environ.get('DATABASE_URL', app.config['database.url'])

# Access config values
is_debug = app.config['app.debug']
db_url = app.config['database.url']
```

## 3. API Design and Routing

Design RESTful APIs. Use Pydantic for request/response validation. Return JSON consistently.

**❌ BAD: Inconsistent routes, no validation, manual JSON**

```python
# ❌ BAD
@app.route('/user/add', method='POST') # Not RESTful
def add_user():
    name = request.forms.get('name') # No validation
    email = request.forms.get('email')
    if not name or not email:
        return '{"error": "Missing fields"}' # Manual JSON string
    # ... logic ...
    return '{"status": "success"}'
```

**✅ GOOD: RESTful routes, Pydantic validation, automatic JSON**

```python
# my_bottle_app/schemas/user_schema.py (✅ GOOD)
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    name: str
    email: EmailStr

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
```

```python
# my_bottle_app/routes/users.py (snippet) (✅ GOOD)
from bottle import Bottle, request, response, HTTPResponse
from pydantic import ValidationError
from my_bottle_app.schemas.user_schema import UserCreate, UserResponse

app = Bottle()

@app.post('/')
def create_user() -> UserResponse:
    try:
        user_data = UserCreate.model_validate(request.json)
    except ValidationError as e:
        raise HTTPResponse(status=400, body={'error': e.errors()}) # Structured error

    # ... call service with user_data.model_dump() ...
    new_user_dict = {'id': 1, 'name': user_data.name, 'email': user_data.email}
    response.status = 201
    return UserResponse.model_validate(new_user_dict).model_dump() # Pydantic handles serialization
```

## 4. Dependency Management (Plugins)

Use Bottle plugins for managing shared resources like database connections. Avoid global mutable state.

**❌ BAD: Global database connection**

```python
# ❌ BAD
import sqlite3
DB_CONN = sqlite3.connect(':memory:') # Global mutable state

@app.route('/data')
def get_data():
    cursor = DB_CONN.cursor() # Uses global
    # ...
```

**✅ GOOD: Plugin-based dependency injection**

```python
# my_bottle_app/plugins/db.py (✅ GOOD)
import sqlite3
from bottle import Plugin, request, abort

class DatabasePlugin(Plugin):
    name = 'database'
    api = 2

    def __init__(self, db_url: str):
        self.db_url = db_url

    def apply(self, callback, route):
        # Only apply to routes that need a DB
        if 'db' not in route.config:
            return callback

        def wrapper(*args, **kwargs):
            try:
                request.db = sqlite3.connect(self.db_url)
                result = callback(*args, **kwargs)
                request.db.commit()
            except sqlite3.Error:
                request.db.rollback()
                abort(500, "Database error")
            finally:
                request.db.close()
            return result
        return wrapper

# my_bottle_app/app.py (snippet)
from my_bottle_app.plugins.db import DatabasePlugin
app.install(DatabasePlugin(app.config['database.url']))

# my_bottle_app/routes/users.py (snippet)
@app.get('/', db=True) # Mark route to use the plugin
def get_all_users() -> List[UserResponse]:
    cursor = request.db.cursor() # Access via request context
    # ...
```

## 5. Type Hints

Apply type hints consistently across all functions, methods, and variables. This improves readability, enables static analysis, and enhances IDE support.

**❌ BAD: Untyped functions**

```python
# ❌ BAD
def process_data(data, user_id):
    # ... no types, hard to understand inputs/outputs ...
    return {"status": "ok"}
```

**✅ GOOD: Fully typed functions**

```python
# my_bottle_app/services/user_service.py (✅ GOOD)
import sqlite3
from typing import Dict, Any, List
from my_bottle_app.schemas.user_schema import UserCreate, UserResponse

class UserService:
    def __init__(self, db_conn: sqlite3.Connection):
        self.db_conn = db_conn

    def get_all(self) -> List[Dict[str, Any]]:
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT id, name, email FROM users")
        return [dict(zip(['id', 'name', 'email'], row)) for row in cursor.fetchall()]

    def create(self, user_data: UserCreate) -> Dict[str, Any]:
        cursor = self.db_conn.cursor()
        cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                       (user_data.name, user_data.email))
        self.db_conn.commit()
        return {'id': cursor.lastrowid, 'name': user_data.name, 'email': user_data.email}
```

## 6. Error Handling

Implement custom error pages for common HTTP errors (400, 404, 500). Raise `HTTPResponse` for controlled error scenarios. Use structured logging for unhandled exceptions.

**❌ BAD: Generic exceptions, no custom error pages**

```python
# ❌ BAD
@app.route('/item/<id>')
def get_item(id):
    if not item_exists(id):
        raise Exception("Item not found") # Generic exception, 500 status
    # ...
```

**✅ GOOD: Custom error handlers, `HTTPResponse`**

```python
# my_bottle_app/app.py (snippet) (✅ GOOD)
import json
from bottle import Bottle, HTTPResponse, request

app = Bottle()

@app.error(404)
def error404(error):
    response.content_type = 'application/json'
    return json.dumps({'error': 'Not Found', 'message': error.body})

@app.error(400)
def error400(error):
    response.content_type = 'application/json'
    return json.dumps({'error': 'Bad Request', 'message': error.body})

@app.error(500)
def error500(error):
    # Log the full traceback for debugging
    app.logger.exception(f"Unhandled error: {error.exception}")
    response.content_type = 'application/json'
    return json.dumps({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'})

# In a route handler:
from bottle import HTTPResponse
def get_resource(resource_id: int):
    if not resource_exists(resource_id):
        raise HTTPResponse(status=404, body={'error': 'Resource not found'}) # Controlled 404
    # ...
```

## 7. Security Best Practices

Validate all input, sanitize output, and use HTTPS. Implement robust authentication/authorization.

**❌ BAD: Direct input use, no output sanitization**

```python
# ❌ BAD
@app.get('/search')
def search():
    query = request.query.get('q') # Direct use of user input
    # ... database query with 'query' (SQL injection risk) ...
    return f"<h1>Search results for {query}</h1>" # XSS risk
```

**✅ GOOD: Pydantic validation, `html_escape`**

```python
# my_bottle_app/schemas/search_schema.py (✅ GOOD)
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    q: str = Field(min_length=1, max_length=100)

# my_bottle_app/routes/search.py (✅ GOOD)
from bottle import Bottle, request, HTTPResponse, html_escape
from pydantic import ValidationError
from my_bottle_app.schemas.search_schema import SearchQuery

app = Bottle()

@app.get('/search')
def search_items():
    try:
        search_params = SearchQuery.model_validate(request.query)
    except ValidationError as e:
        raise HTTPResponse(status=400, body={'error': e.errors()})

    sanitized_query = search_params.q # Input is now validated
    # ... safe database query using parameters ...
    results = ["Item A", "Item B"] # Example results

    # Sanitize output before rendering
    escaped_query = html_escape(sanitized_query)
    return f"<h1>Search results for {escaped_query}</h1><p>Found: {', '.join(html_escape(r) for r in results)}</p>"
```

## 8. Performance Considerations

Keep route handlers lean. Delegate complex business logic and I/O operations to service layers or background tasks.

**❌ BAD: Heavy computation in a route handler**

```python
# ❌ BAD
@app.post('/report')
def generate_report():
    # This will block the web server for a long time
    large_data = fetch_huge_dataset()
    processed_data = perform_complex_analytics(large_data)
    save_report_to_db(processed_data)
    return {'status': 'Report generated'}
```

**✅ GOOD: Delegate to a service or background task**

```python
# my_bottle_app/services/report_service.py (✅ GOOD)
import threading
import time

def _generate_report_async(data: Dict[str, Any]):
    # Simulate heavy work
    time.sleep(5)
    print(f"Report generated for {data['user_id']}")

class ReportService:
    def generate_report_in_background(self, user_id: int):
        # Use a thread or a proper task queue (Celery, RQ) for production
        report_data = {'user_id': user_id, 'timestamp': time.time()}
        thread = threading.Thread(target=_generate_report_async, args=(report_data,))
        thread.start()
        return {"message": "Report generation started in background."}

# my_bottle_app/routes/reports.py (✅ GOOD)
from bottle import Bottle, request
from my_bottle_app.services.report_service import ReportService

app = Bottle()
report_service = ReportService() # Instantiate service

@app.post('/reports')
def create_report():
    user_id = request.json.get('user_id')
    if not user_id:
        raise HTTPResponse(status=400, body={'error': 'user_id is required'})
    
    result = report_service.generate_report_in_background(user_id)
    return result
```

## 9. Testing

Write unit and integration tests using `pytest` and `webtest`.

**❌ BAD: No automated tests**

```python
# ❌ BAD
# No test files, relying on manual browser checks.
```

**✅ GOOD: `pytest` and `webtest` for route testing**

```python
# my_bottle_app/tests/test_users.py (✅ GOOD)
import pytest
from webtest import TestApp
from my_bottle_app.app import app as main_app # Import your main Bottle app

@pytest.fixture
def test_app():
    """Provides a WebTest client for the Bottle application."""
    return TestApp(main_app)

def test_get_all_users(test_app):
    """Test that the /users endpoint returns a list."""
    resp = test_app.get('/users')
    assert resp.status_code == 200
    assert isinstance(resp.json, list)
    # Add more specific assertions for content if needed

def test_create_user_success(test_app):
    """Test creating a user successfully."""
    user_data = {"name": "Test User", "email": "test@example.com"}
    resp = test_app.post_json('/users', params=user_data)
    assert resp.status_code == 201
    assert resp.json['name'] == "Test User"
    assert resp.json['email'] == "test@example.com"
    assert 'id' in resp.json

def test_create_user_invalid_data(test_app):
    """Test creating a user with invalid data."""
    invalid_data = {"name": "Invalid"} # Missing email
    resp = test_app.post_json('/users', params=invalid_data, expect_errors=True)
    assert resp.status_code == 400
    assert 'error' in resp.json
```