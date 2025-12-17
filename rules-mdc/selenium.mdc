---
description: Definitive guidelines for writing robust, maintainable, and readable Selenium automation scripts in Python, focusing on modern best practices for testing and web scraping.
globs: **/*.py
---
# selenium Best Practices

This guide outlines the definitive best practices for using Selenium with Python. Adhere to these rules to ensure your automation scripts are robust, maintainable, and performant.

## 1. Code Organization: Page Object Model (POM)

Always structure your Selenium code using the Page Object Model (POM). This separates UI elements and interactions from your test or scraping logic, making your code highly maintainable and reusable.

❌ **BAD: Logic and locators mixed**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://example.com/login")
driver.find_element(By.ID, "username").send_keys("user")
driver.find_element(By.ID, "password").send_keys("pass")
driver.find_element(By.XPATH, "//button[text()='Login']").click()
# ... more logic
```

✅ **GOOD: Page Object Model**
```python
# pages/base_page.py
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class BasePage:
    def __init__(self, driver: WebDriver, timeout: int = 10):
        self.driver = driver
        self.wait = WebDriverWait(driver, timeout)

    def _find_element(self, by: By, value: str):
        return self.wait.until(EC.presence_of_element_located((by, value)))

# pages/login_page.py
from pages.base_page import BasePage
from selenium.webdriver.common.by import By

class LoginPage(BasePage):
    URL = "https://example.com/login"
    _USERNAME_INPUT = (By.ID, "username")
    _PASSWORD_INPUT = (By.ID, "password")
    _LOGIN_BUTTON = (By.XPATH, "//button[text()='Login']")

    def open(self):
        self.driver.get(self.URL)

    def login(self, username, password):
        self._find_element(*self._USERNAME_INPUT).send_keys(username)
        self._find_element(*self._PASSWORD_INPUT).send_keys(password)
        self._find_element(*self._LOGIN_BUTTON).click()

# tests/test_login.py (or main scraping script)
from selenium import webdriver
from pages.login_page import LoginPage

def test_successful_login():
    driver = webdriver.Chrome()
    try:
        login_page = LoginPage(driver)
        login_page.open()
        login_page.login("valid_user", "valid_pass")
        assert "dashboard" in driver.current_url # Example assertion
    finally:
        driver.quit()
```

## 2. Robustness: Explicit Waits

Never use `time.sleep()` or rely solely on implicit waits. Use `WebDriverWait` with `expected_conditions` to explicitly wait for elements to be in a specific state. This prevents flaky tests and ensures scripts are resilient to dynamic page loading.

❌ **BAD: Arbitrary waits or implicit waits**
```python
import time
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By

driver: WebDriver = ...
driver.implicitly_wait(10) # Still prone to issues
driver.get("https://example.com")
time.sleep(5) # NEVER DO THIS
element = driver.find_element(By.ID, "dynamic_content")
```

✅ **GOOD: Explicit Waits with Expected Conditions**
```python
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

driver: WebDriver = ...
driver.get("https://example.com")
wait = WebDriverWait(driver, 10)

# Wait for element to be visible
dynamic_element = wait.until(EC.visibility_of_element_located((By.ID, "dynamic_content")))

# Wait for element to be clickable
button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".submit-btn")))
button.click()
```

## 3. Driver Management & Headless Execution

Always use Selenium Manager (built into Selenium 4.6+) or `webdriver-manager` to automatically handle browser driver binaries. Run browsers in headless mode for performance and CI/CD environments.

❌ **BAD: Manual driver downloads, visible browser**
```python
from selenium import webdriver
# Requires chromedriver.exe in PATH or specified manually
driver = webdriver.Chrome(executable_path="/path/to/chromedriver")
driver.get("https://example.com") # Opens a visible browser window
```

✅ **GOOD: Automated driver management, headless with optimized options**
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def get_headless_chrome_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")  # Modern headless mode
    options.add_argument("--no-sandbox")    # Essential for CI/Docker
    options.add_argument("--disable-dev-shm-usage") # Avoids /dev/shm issues
    options.add_argument("--window-size=1920,1080") # Ensure consistent rendering
    options.add_argument("--disable-gpu") # Recommended for headless
    options.add_argument("--log-level=3") # Suppress verbose logging
    return webdriver.Chrome(options=options)

driver = get_headless_chrome_driver()
try:
    driver.get("https://example.com")
    print(driver.title)
finally:
    driver.quit() # ALWAYS quit the driver
```

## 4. Locator Strategy Prioritization

Choose locators that are robust and least likely to change. Prioritize `By.ID` and `By.CSS_SELECTOR`. Avoid fragile `By.XPATH` unless absolutely necessary, and `By.CLASS_NAME` for elements with multiple classes.

❌ **BAD: Fragile Locators**
```python
# Absolute XPath (breaks with any DOM change)
element = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/form/input[3]")
# Class name for multiple classes (ambiguous)
element = driver.find_element(By.CLASS_NAME, "btn btn-primary active")
```

✅ **GOOD: Robust Locators**
```python
# Unique ID (most reliable)
element = driver.find_element(By.ID, "submitButton")
# CSS Selector (flexible, powerful, and stable)
element = driver.find_element(By.CSS_SELECTOR, "button.submit-btn[name='action']")
# Partial Link Text (for navigation)
element = driver.find_element(By.PARTIAL_LINK_TEXT, "Read More")
```

## 5. Error Handling & Teardown

Implement robust `try...finally` blocks to ensure `driver.quit()` is always called, preventing lingering browser processes. Catch common Selenium exceptions for graceful failure and better debugging.

```python
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException

driver = get_headless_chrome_driver() # Using the good practice from above
try:
    driver.get("https://example.com/nonexistent")
    # Attempt to find an element that might not exist
    element = driver.find_element(By.ID, "some_element")
    element.click()
except NoSuchElementException:
    print("Element not found. Check locator or page state.")
except TimeoutException:
    print("Page load or element wait timed out.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    driver.quit() # Ensures browser process is terminated
```

## 6. Type Hints

Always use Python type hints. They improve code readability, enable static analysis, and reduce bugs, especially in larger projects and Page Object Models.

```python
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.common.by import By
from typing import Tuple

class MyPage:
    _SEARCH_INPUT: Tuple[By, str] = (By.ID, "search")

    def __init__(self, driver: WebDriver):
        self.driver = driver

    def search(self, query: str) -> None:
        search_box: WebElement = self.driver.find_element(*self._SEARCH_INPUT)
        search_box.send_keys(query)
        search_box.submit()
```

## 7. Virtual Environments & Packaging

Always use a virtual environment (`venv` or `uv`) for dependency management. For modern projects, use `uv` and `pyproject.toml` for fast, reproducible builds.

```bash
# Recommended modern setup with uv
uv init my-selenium-project
cd my-selenium-project
uv add selenium pytest # Add your dependencies

# Or classic venv
python -m venv .venv
source .venv/bin/activate # Linux/macOS
.venv\Scripts\Activate.ps1 # Windows PowerShell
pip install selenium pytest
```