---
description: This guide outlines definitive best practices for using the `requests` library in Python, focusing on performance, reliability, and maintainability for API clients and web interactions.
globs: **/*.py
---
# `requests` Best Practices

`requests` is the definitive HTTP client for Python. To build robust, performant, and maintainable web interactions, you must leverage its advanced features and integrate them with modern Python best practices. This guide provides actionable rules for our team.

## 1. Always Use `Session` Objects

For any code making more than a single HTTP call, or any reusable API client, you **must** use a `requests.Session` object. Sessions provide connection pooling and cookie persistence, drastically improving performance and resource usage.

❌ **BAD: Direct `requests` calls**
```python
import requests

# Each call creates a new connection
response1 = requests.get("https://api.example.com/data/1")
response2 = requests.get("https://api.example.com/data/2")
```

✅ **GOOD: Use a `Session` with a context manager**
```python
import requests

with requests.Session() as session:
    # Connections are pooled and reused
    response1 = session.get("https://api.example.com/data/1")
    response2 = session.get("https://api.example.com/data/2")
```

## 2. Configure Retries and Timeouts with `HTTPAdapter`

Enhance session reliability by mounting a custom `HTTPAdapter` to handle retries with backoff and set default timeouts. This prevents flaky network issues from crashing your application and ensures requests don't hang indefinitely.

❌ **BAD: No retries, no default timeouts**
```python
import requests

with requests.Session() as session:
    # Will fail on first network glitch, can hang forever
    response = session.get("https://api.example.com/flaky-endpoint")
```

✅ **GOOD: Mount an `HTTPAdapter` with `Retry` logic**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure retry strategy
retry_strategy = Retry(
    total=3,  # Total number of retries
    backoff_factor=1,  # Exponential backoff (1, 2, 4 seconds)
    status_forcelist=[429, 500, 502, 503, 504], # HTTP statuses to retry on
    allowed_methods=["HEAD", "GET", "OPTIONS"] # Methods to retry
)
adapter = HTTPAdapter(max_retries=retry_strategy)

with requests.Session() as session:
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # All requests made with this session will use the retry logic
    response = session.get("https://api.example.com/flaky-endpoint", timeout=(5, 10))
```
**Note:** `timeout` should still be explicitly set on individual requests, even with an adapter, to override or confirm the default.

## 3. Always Set Explicit Timeouts

Never make a `requests` call without an explicit `timeout` parameter. This prevents your application from hanging indefinitely due to slow or unresponsive servers. Specify a tuple `(connect_timeout, read_timeout)`.

*   `connect_timeout`: The time limit for the client to establish a connection to the server.
*   `read_timeout`: The time limit for the client to wait for a response after sending the request.

❌ **BAD: Missing timeout**
```python
response = session.get("https://api.example.com/slow-endpoint") # Can hang forever
```

✅ **GOOD: Explicit `(connect, read)` timeout**
```python
# Wait 5 seconds to connect, 10 seconds to receive data
response = session.get("https://api.example.com/slow-endpoint", timeout=(5, 10))
```

## 4. Handle Responses Robustly

Process responses defensively. Always check for HTTP status codes, handle JSON parsing errors, and log sufficient context for debugging.

❌ **BAD: Naive response handling**
```python
response = session.get("https://api.example.com/data")
data = response.json() # Fails if not JSON or 204 No Content
print(data) # Continues even if HTTP 4xx/5xx
```

✅ **GOOD: `raise_for_status()` and `try/except` for JSON**
```python
import logging
import requests

logging.basicConfig(level=logging.INFO)

try:
    response = session.get("https://api.example.com/data", timeout=(5, 10))
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses

    try:
        data = response.json()
        logging.info(f"Successfully fetched data from {response.url}")
        # Process data
    except requests.exceptions.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from {response.url}: {e}")
        logging.error(f"Response content: {response.text[:200]}") # Log partial content
        raise # Re-raise or handle appropriately

except requests.exceptions.HTTPError as e:
    logging.error(f"HTTP Error for {e.request.url}: {e.response.status_code} - {e.response.text}")
    raise
except requests.exceptions.ConnectionError as e:
    logging.error(f"Connection Error for {e.request.url}: {e}")
    raise
except requests.exceptions.Timeout as e:
    logging.error(f"Timeout Error for {e.request.url}: {e}")
    raise
except requests.exceptions.RequestException as e:
    logging.error(f"An unexpected error occurred during request to {e.request.url}: {e}")
    raise
```

## 5. Encapsulate Client Logic

Isolate all `requests` logic within a dedicated API client class or module. This improves testability, reusability, and maintainability. Use type hints for clarity.

❌ **BAD: Scattered `requests` calls**
```python
# In main.py
def process_user_data(user_id):
    resp = requests.get(f"https://api.example.com/users/{user_id}")
    # ... more logic ...

# In another_module.py
def fetch_product_info(product_id):
    resp = requests.get(f"https://api.example.com/products/{product_id}")
    # ... more logic ...
```

✅ **GOOD: Dedicated API Client Class**
```python
# api_client.py
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class ExampleAPIClient:
    """
    Client for interacting with the Example API.
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        session: Optional[requests.Session] = None,
        timeout: Tuple[float, float] = (5, 10)
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

        if session is None:
            self.session = requests.Session()
            self._configure_session()
        else:
            self.session = session

    def _configure_session(self) -> None:
        """Configures the session with adapters and default headers."""
        retry_strategy = Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Internal helper for making API requests."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.request(
                method, url, params=params, json=json, data=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON decode error from {url}: {e}. Response: {response.text[:200]}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url} ({method}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status: {e.response.status_code}, Body: {e.response.text[:200]}")
            raise

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Fetches user details by ID."""
        return self._request("GET", f"users/{user_id}")

    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new user."""
        return self._request("POST", "users", json=user_data)

# In main.py
from api_client import ExampleAPIClient
import os

# Use environment variables for sensitive info and configuration
API_BASE_URL = os.getenv("EXAMPLE_API_BASE_URL", "https://default.api.example.com/v1")
API_KEY = os.getenv("EXAMPLE_API_KEY", "your_fallback_key")

try:
    client = ExampleAPIClient(base_url=API_BASE_URL, api_key=API_KEY)
    user = client.get_user("123")
    print(f"Fetched user: {user}")

    new_user = client.create_user({"name": "Jane Doe", "email": "jane@example.com"})
    print(f"Created user: {new_user}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

## 6. Avoid Hardcoding URLs and Credentials

Externalize base URLs, API keys, and other sensitive configurations using environment variables or dedicated configuration files. This is crucial for security and deploying across different environments (dev, staging, prod).

❌ **BAD: Hardcoded values**
```python
# In code
API_URL = "https://prod.api.example.com/v1"
AUTH_TOKEN = "super_secret_token_123"
```

✅ **GOOD: Environment variables**
```python
# In code
import os
API_URL = os.getenv("EXAMPLE_API_URL", "https://dev.api.example.com/v1")
AUTH_TOKEN = os.getenv("EXAMPLE_API_TOKEN")

if not AUTH_TOKEN:
    raise ValueError("EXAMPLE_API_TOKEN environment variable not set.")
```

## 7. Test Your Client Code

When testing your API client, mock the `requests` library to avoid making actual network calls. This makes tests fast, reliable, and independent of external service availability. Libraries like `responses` or `unittest.mock` are excellent for this.

```python
# test_api_client.py
import unittest
from unittest.mock import patch, MagicMock
from api_client import ExampleAPIClient
import requests

class TestExampleAPIClient(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock(spec=requests.Session)
        self.client = ExampleAPIClient(
            base_url="http://test.api.com", api_key="test_key", session=self.mock_session
        )

    def test_get_user_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "name": "Test User"}
        mock_response.raise_for_status.return_value = None # No HTTPError

        self.mock_session.request.return_value = mock_response

        user = self.client.get_user("123")
        self.assertEqual(user["name"], "Test User")
        self.mock_session.request.assert_called_once_with(
            "GET", "http://test.api.com/users/123",
            params=None, json=None, data=None, headers=None, timeout=(5, 10)
        )

    def test_get_user_http_error(self):
        self.mock_session.request.side_effect = requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: http://test.api.com/users/999"
        )
        with self.assertRaises(requests.exceptions.HTTPError):
            self.client.get_user("999")

if __name__ == "__main__":
    unittest.main()
```