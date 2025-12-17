---
description: This guide provides definitive, opinionated best practices for building robust, scalable, and maintainable Flask applications, emphasizing modern patterns and common pitfalls.
globs: **/*.py
---
# Flask Best Practices

Flask is a powerful microframework. To leverage it effectively for scalable, maintainable applications, adhere to these modern best practices.

## 1. Code Organization: Application Factories & Blueprints

Always structure your Flask application as a package using an **application factory** and **Blueprints**. This pattern is crucial for testability, multiple environments, and modularity.

### 1.1 Application Factory

Use an application factory to create your Flask app instance. This isolates configuration and extensions, preventing global state issues.

❌ **BAD**: Global `app` instance
```python
# app.py
from flask import Flask
app = Flask(__name__) # Global app instance
# ... configure app, register routes directly
```

✅ **GOOD**: Application factory
```python
# myapp/__init__.py
from flask import Flask
from .config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions here, e.g., db.init_app(app)
    # Register blueprints here, e.g., app.register_blueprint(main_bp)

    return app

# run.py (for development/entrypoint)
from myapp import create_app
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
```

### 1.2 Blueprints for Modularity

Group related routes, models, and services into Blueprints. This keeps your application organized and promotes reusability.

```python
# myapp/auth/routes.py
from flask import Blueprint, render_template, request, redirect, url_for
from . import auth_bp # Import the blueprint instance

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # ... authentication logic
        return redirect(url_for('main.index'))
    return render_template('auth/login.html')

# myapp/auth/__init__.py
from flask import Blueprint
auth_bp = Blueprint('auth', __name__, template_folder='templates')
from . import routes # Import routes to register them with the blueprint

# myapp/__init__.py (inside create_app)
from myapp.auth import auth_bp
app.register_blueprint(auth_bp, url_prefix='/auth')
```

## 2. Configuration Management

Manage configuration external to your code. Prioritize environment variables for sensitive data and dynamic settings.

✅ **GOOD**: Environment variables and `Config` class
```python
# myapp/config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-very-secret-key-fallback'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
    DEBUG = os.environ.get('FLASK_DEBUG') == '1'

# .env (for local development, not committed to VCS)
SECRET_KEY=my-super-secret-key
DATABASE_URL=postgresql://user:password@host:port/dbname
FLASK_DEBUG=1
```

## 3. Performance Considerations

Build fast backends by leveraging asynchronous patterns and efficient deployment.

### 3.1 Asynchronous View Functions

For I/O-bound tasks (database calls, external API requests), use `async def` views with an ASGI server.

❌ **BAD**: Blocking I/O in sync view
```python
# app.py
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/sync_data')
def sync_data():
    time.sleep(2) # Simulates blocking I/O
    return jsonify({"message": "Data fetched synchronously"})
```

✅ **GOOD**: Non-blocking I/O with `async def`
```python
# myapp/api/routes.py
from flask import Blueprint, jsonify
import asyncio # For async operations
# For actual async DB/HTTP, use aiohttp, asyncpg, etc.

api_bp = Blueprint('api', __name__)

@api_bp.route('/async_data')
async def async_data():
    await asyncio.sleep(2) # Simulates non-blocking I/O
    return jsonify({"message": "Data fetched asynchronously"})

# Deploy with an ASGI server like Uvicorn or Hypercorn:
# uvicorn myapp:create_app --factory --port 8000
```

### 3.2 WSGI/ASGI Server

Never use Flask's built-in development server in production. Use a robust WSGI (Gunicorn, uWSGI) or ASGI (Uvicorn, Hypercorn) server.

```bash
# For WSGI (sync Flask apps)
gunicorn -w 4 'myapp:create_app()' --bind 0.0.0.0:8000 --timeout 60

# For ASGI (async Flask apps, requires Flask 2.0+ and an ASGI server)
uvicorn myapp:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

## 4. Security Best Practices

Security is paramount. Implement these measures from the start.

### 4.1 Protect Secrets

Store all sensitive information (API keys, database credentials) in environment variables or a secure secret management system. Never hardcode them.

❌ **BAD**: Hardcoded secret
```python
# app.py
DB_PASSWORD = "my_hardcoded_password"
```

✅ **GOOD**: Environment variable
```python
# myapp/config.py
import os
DB_PASSWORD = os.environ.get('DB_PASSWORD')
if not DB_PASSWORD:
    raise ValueError("No DB_PASSWORD set for production")
```

### 4.2 Input Validation

Validate all user input rigorously. Use a library like WTForms or Pydantic for forms and API payloads.

```python
# myapp/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Email, Length

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])

# myapp/auth/routes.py
from .forms import LoginForm

@auth_bp.route('/login', methods=['POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        # Process valid data
        email = form.email.data
        password = form.password.data
        # ...
    else:
        # Handle invalid input, e.g., flash errors
        pass
```

### 4.3 CSRF Protection

Enable CSRF protection for all forms that modify state. Use Flask-WTF which integrates WTForms with CSRF.

```python
# myapp/__init__.py (inside create_app)
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect()
csrf.init_app(app)

# In your forms, Flask-WTF automatically adds a hidden CSRF token field.
# In templates, ensure you render it:
# <form method="POST">
#     {{ form.csrf_token }}
#     ...
# </form>
```

## 5. Error Handling

Provide graceful error handling and detailed logging.

### 5.1 Custom Error Pages

Register custom error handlers for common HTTP errors.

```python
# myapp/errors/handlers.py
from flask import Blueprint, render_template

errors_bp = Blueprint('errors', __name__)

@errors_bp.app_errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@errors_bp.app_errorhandler(500)
def internal_error(error):
    # Log the error for debugging
    return render_template('errors/500.html'), 500

# myapp/__init__.py (inside create_app)
from myapp.errors import errors_bp
app.register_blueprint(errors_bp)
```

## 6. API Design: RESTful Principles & Type Hints

Design APIs following RESTful principles, return JSON, and use type hints.

### 6.1 RESTful Endpoints & JSON Responses

Use appropriate HTTP methods and return JSON for API endpoints.

❌ **BAD**: Inconsistent methods, HTML response for API
```python
@api_bp.route('/users/get_all', methods=['POST']) # Should be GET
def get_users_html():
    return "<h1>All Users</h1>" # Should be JSON
```

✅ **GOOD**: RESTful, JSON response
```python
from flask import jsonify, request

@api_bp.route('/users', methods=['GET'])
def get_users():
    users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    return jsonify(users)

@api_bp.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    # ... create user with data
    return jsonify({"message": "User created", "id": 3}), 201
```

### 6.2 Type Hints

Mandatory for all new Python code. Improves readability, maintainability, and enables static analysis.

❌ **BAD**: Untyped function
```python
def calculate_total(price, quantity):
    return price * quantity
```

✅ **GOOD**: Type-hinted function
```python
def calculate_total(price: float, quantity: int) -> float:
    return price * quantity
```

## 7. Pallets Python Styleguide

Adhere to the [Pallets Python Styleguide](https://palletsprojects.com/governance/sourcecode/python-styleguide/) for consistent code formatting and structure. This aligns with PEP 8 but includes Flask-specific conventions. Use linters (e.g., Black, Flake8) and static analysis (e.g., MyPy) to enforce this automatically.