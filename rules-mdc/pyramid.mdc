---
description: This guide provides opinionated, actionable best practices for building modern, maintainable, and performant applications with the Pyramid web framework.
globs: **/*.py
---
# pyramid Best Practices

Pyramid is a flexible, "pay-only-what-you-use" Python web framework. To build robust and scalable Pyramid applications, adhere to these modern best practices.

## 1. Code Organization and Structure

Organize your application for clarity, testability, and scalability. Avoid monolithic `__init__.py` files.

### 1.1. Enforce Code Style with `black` and `flake8`

Always format your code automatically and lint it. Integrate `black` and `flake8` with `pre-commit` hooks.

**❌ BAD - Inconsistent Formatting & Linting**
```python
# Inconsistent spacing, line length, and missing imports
from pyramid.view import view_config
import os
def my_view(request):
    data = request.json_body
    if data and 'name' in data:
        return {'status': 'ok', 'message': f"Hello, {data['name']}!"}
    return {'status': 'error', 'message': 'Name is required.'}
```

**✅ GOOD - Automated Formatting & Linting**
```python
# Ensure your pre-commit setup handles this automatically
# Example .pre-commit-config.yaml:
# - repo: https://github.com/psf/black
#   rev: 23.1.0
#   hooks: [- id: black]
# - repo: https://github.com/PyCQA/flake8
#   rev: 6.0.0
#   hooks: [- id: flake8]

from pyramid.view import view_config
from pyramid.response import Response
from typing import Dict, Any

@view_config(route_name="hello", renderer="json")
def my_view(request) -> Dict[str, str]:
    """Handles a greeting request."""
    data: Dict[str, Any] = request.json_body
    if data and "name" in data:
        return {"status": "ok", "message": f"Hello, {data['name']}!"}
    return Response(json_body={"status": "error", "message": "Name is required."}, status=400)
```

### 1.2. Modularize Configuration

Separate your application's configuration (routes, views, security) into distinct modules. Use `config.include()` to compose your application.

**❌ BAD - Monolithic `__init__.py`**
```python
# myapp/__init__.py
from pyramid.config import Configurator
from pyramid.view import view_config

def main(global_config, **settings):
    with Configurator(settings=settings) as config:
        config.add_route('home', '/')
        config.add_route('users', '/users')
        config.add_route('user_detail', '/users/{id}')

        @view_config(route_name='home', renderer='json')
        def home_view(request):
            return {'project': 'myapp'}

        # ... hundreds of lines of routes and views ...
    return config.make_wsgi_app()
```

**✅ GOOD - Modular Configuration with `include()`**
```python
# myapp/__init__.py
from pyramid.config import Configurator

def main(global_config, **settings):
    with Configurator(settings=settings) as config:
        config.include("pyramid_jinja2") # Example external package
        config.include(".routes")        # Local routes
        config.include(".views")         # Local views (for decorator-based views)
        config.include(".security")      # Local security policies
        config.scan(".")                 # Scan for decorator-based views and subscribers
    return config.make_wsgi_app()

# myapp/routes.py
def includeme(config):
    config.add_route("home", "/")
    config.add_route("users", "/users")
    config.add_route("user_detail", "/users/{id}")

# myapp/views.py (or specific modules like myapp/views/user.py)
from pyramid.view import view_config
from pyramid.response import Response
from typing import Dict, Any

@view_config(route_name="home", renderer="json")
def home_view(request) -> Dict[str, str]:
    return {"project": "myapp"}

@view_config(route_name="users", request_method="GET", renderer="json")
def list_users_view(request) -> Dict[str, Any]:
    # ... logic to list users ...
    return {"users": []}
```

## 2. Common Patterns and Anti-patterns

Leverage Pyramid's design principles to write clean, maintainable code.

### 2.1. Use View Predicates for Conditional Logic

Avoid `if/else` statements within views to handle different request methods, authentication states, or other request properties. Use `@view_config` predicates instead.

**❌ BAD - Logic in View**
```python
from pyramid.view import view_config
from pyramid.response import Response

@view_config(route_name="items", renderer="json")
def items_view(request):
    if request.method == "GET":
        return {"items": ["item1", "item2"]}
    elif request.method == "POST":
        # ... create item logic ...
        return Response(json_body={"message": "Item created"}, status=201)
    else:
        return Response(status=405) # Method Not Allowed
```

**✅ GOOD - Predicates for Clarity**
```python
from pyramid.view import view_config
from pyramid.response import Response
from typing import Dict, Any

@view_config(route_name="items", request_method="GET", renderer="json")
def get_items_view(request) -> Dict[str, Any]:
    """Retrieves a list of items."""
    return {"items": ["item1", "item2"]}

@view_config(route_name="items", request_method="POST", renderer="json")
def create_item_view(request) -> Response:
    """Creates a new item."""
    # ... create item logic ...
    return Response(json_body={"message": "Item created"}, status=201)
```

### 2.2. Embrace Pyramid's Transaction Management

Utilize the transaction management system provided by Pyramid's cookiecutters (e.g., `pyramid_tm`). This ensures atomicity for database operations without manual `commit`/`rollback` in your view logic.

**❌ BAD - Manual Transaction Handling**
```python
from pyramid.view import view_config
from myapp.models import DBSession, User

@view_config(route_name="register", request_method="POST", renderer="json")
def register_user_view(request):
    try:
        user = User(name=request.json_body["name"])
        DBSession.add(user)
        DBSession.flush() # Manual flush
        DBSession.commit() # Manual commit
        return {"message": "User registered", "id": user.id}
    except Exception:
        DBSession.rollback() # Manual rollback
        raise
```

**✅ GOOD - Automatic Transaction Handling (with `pyramid_tm`)**
```python
# Ensure pyramid_tm is configured in your __init__.py
# config.include('pyramid_tm')

from pyramid.view import view_config
from myapp.models import DBSession, User
from typing import Dict, Any

@view_config(route_name="register", request_method="POST", renderer="json")
def register_user_view(request) -> Dict[str, Any]:
    """Registers a new user, transaction handled automatically."""
    user = User(name=request.json_body["name"])
    DBSession.add(user)
    DBSession.flush() # Flush is fine, commit/rollback are handled by pyramid_tm
    return {"message": "User registered", "id": user.id}
```

### 2.3. Avoid Mutable Global State and Singletons

Pyramid is designed to run multiple application instances in a single process. Avoid patterns that introduce mutable global state, which can lead to hard-to-debug side effects.

**❌ BAD - Global State**
```python
# myapp/settings.py (BAD practice)
GLOBAL_CONFIG = {} # Mutable global dictionary

def initialize_config(app_settings):
    GLOBAL_CONFIG.update(app_settings)

# myapp/views.py
from myapp.settings import GLOBAL_CONFIG

@view_config(route_name="config_info", renderer="json")
def config_info_view(request):
    return {"global_setting": GLOBAL_CONFIG.get("some_setting")}
```

**✅ GOOD - Pass Configuration via `request.registry.settings`**
```python
# myapp/__init__.py
from pyramid.config import Configurator

def main(global_config, **settings):
    with Configurator(settings=settings) as config:
        # settings are automatically available via request.registry.settings
        pass
    return config.make_wsgi_app()

# myapp/views.py
from pyramid.view import view_config
from typing import Dict, Any

@view_config(route_name="config_info", renderer="json")
def config_info_view(request) -> Dict[str, Any]:
    """Retrieves configuration information."""
    return {"some_setting": request.registry.settings.get("some_setting")}
```

## 3. Type Hints

Always use type hints for functions, methods, and variables. This improves code readability, enables static analysis, and helps prevent runtime errors.

**❌ BAD - Untyped Code**
```python
def calculate_total(price, quantity):
    return price * quantity

class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price
```

**✅ GOOD - Fully Typed Code**
```python
from typing import Union

def calculate_total(price: Union[int, float], quantity: int) -> Union[int, float]:
    """Calculates the total price for a given quantity."""
    return price * quantity

class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price
```

## 4. API Design

Design RESTful APIs with clear resources, HTTP verbs, and consistent response formats.

### 4.1. Consistent JSON Responses

Always return JSON for API endpoints, including error responses. Use `pyramid.response.Response` for custom status codes.

**❌ BAD - Inconsistent Error Responses**
```python
# Sometimes returns HTML, sometimes JSON, inconsistent error structure
@view_config(route_name="data", renderer="json")
def get_data_view(request):
    if not request.authenticated_userid:
        return Response("Unauthorized", status=401) # Returns plain text
    # ...
    return {"data": "some_data"}
```

**✅ GOOD - Consistent JSON Responses**
```python
from pyramid.view import view_config
from pyramid.response import Response
from typing import Dict, Any

@view_config(route_name="data", request_method="GET", renderer="json")
def get_data_view(request) -> Union[Dict[str, Any], Response]:
    """Retrieves data, returns consistent JSON."""
    if not request.authenticated_userid:
        return Response(
            json_body={"error": "Unauthorized", "code": "AUTH_001"},
            status=401,
        )
    # ... logic to fetch data ...
    return {"data": "some_data"}
```