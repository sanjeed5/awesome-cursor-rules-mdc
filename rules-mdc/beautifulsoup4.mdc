---
description: This guide provides opinionated, actionable, and example-driven best practices for using beautifulsoup4 in Python web scraping projects, focusing on modern techniques and common pitfalls.
globs: **/*.py
---
# beautifulsoup4 Best Practices

Beautiful Soup 4 (BS4) is the definitive library for parsing static HTML/XML. Use it as a robust component within a well-structured scraping pipeline.

## 1. Installation & Parser Selection

Always install `beautifulsoup4` alongside `lxml` for optimal performance. `lxml` is significantly faster than Python's built-in `html.parser`. `html5lib` is a useful fallback for extremely malformed HTML.

```bash
pip install beautifulsoup4 lxml requests html5lib
```

**✅ GOOD: Explicitly specify `lxml` as the parser.**
```python
from bs4 import BeautifulSoup

# Always use lxml for speed
soup = BeautifulSoup(html_content, 'lxml')
```

**❌ BAD: Relying on default `html.parser` or `html5lib` unnecessarily.**
```python
# Slower, less efficient
soup = BeautifulSoup(html_content, 'html.parser')
# Only use html5lib if lxml fails on malformed HTML
soup = BeautifulSoup(html_content, 'html5lib')
```

## 2. Ethical Scraping Fundamentals

Respect `robots.txt`, implement rate limiting, and use a `User-Agent`. This is non-negotiable for responsible scraping.

**✅ GOOD: Respect `robots.txt` and use a `User-Agent` with delays.**
```python
import requests
import time
from typing import Dict

def fetch_page_ethically(url: str, headers: Dict[str, str], delay_seconds: float = 2.0) -> str:
    """Fetches HTML content with ethical considerations."""
    # In a real app, check robots.txt programmatically
    # e.g., using 'robotexclusionrulesparser' library
    print(f"Fetching {url}...")
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    time.sleep(delay_seconds) # Rate limiting
    return response.text

# Example usage
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 MyScraper/1.0'
}
# html_content = fetch_page_ethically("https://example.com/products", HEADERS)
```

**❌ BAD: Blindly hitting endpoints without identification or delays.**
```python
# Will likely get blocked or cause issues
response = requests.get("https://example.com/products")
html_content = response.text
```

## 3. Code Organization & Type Hinting

Structure your scraping logic into small, testable, type-hinted functions. Separate concerns: fetching, parsing, and data modeling. Use `pydantic` or `dataclasses` for structured output.

**✅ GOOD: Modular, type-hinted functions with structured data output.**
```python
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel
from typing import List, Optional

class Product(BaseModel):
    name: str
    price: float
    url: str
    in_stock: bool

def parse_product_card(card_tag: Tag) -> Optional[Product]:
    """Parses a single product card HTML tag into a Product model."""
    name_tag = card_tag.select_one('h4.product-name')
    price_tag = card_tag.select_one('.price-wrapper')
    link_tag = card_tag.select_one('a.product-link')
    stock_tag = card_tag.select_one('.stock-status')

    if not all([name_tag, price_tag, link_tag]):
        return None # Skip malformed cards

    try:
        name = name_tag.get_text(strip=True)
        price = float(price_tag.get_text(strip=True).replace('$', ''))
        url = link_tag['href']
        in_stock = "In Stock" in stock_tag.get_text(strip=True) if stock_tag else False
        return Product(name=name, price=price, url=url, in_stock=in_stock)
    except (AttributeError, ValueError, KeyError) as e:
        print(f"Error parsing product card: {e}")
        return None

def extract_products_from_html(html_content: str) -> List[Product]:
    """Extracts all products from a given HTML page."""
    soup = BeautifulSoup(html_content, 'lxml')
    product_cards = soup.select('.product-card') # Use CSS selectors
    products = [
        product for card in product_cards
        if (product := parse_product_card(card)) is not None
    ]
    return products

# Example: products = extract_products_from_html(html_content)
```

**❌ BAD: Monolithic functions, untyped code, unstructured output.**
```python
def scrape_products(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    products = []
    for card in soup.find_all('div', class_='product-card'): # Less readable
        name = card.find('h4').text
        price = float(card.find('span', class_='price').text.replace('$', ''))
        # ... lots of inline logic ...
        products.append({'name': name, 'price': price})
    return products
```

## 4. Parsing Efficiency: CSS Selectors & Text Extraction

Prioritize CSS selectors (`.select()` or `.select_one()`) over `find_all()` for readability, power, and often better performance. Always strip whitespace from extracted text.

**✅ GOOD: CSS selectors and `get_text(strip=True)`.**
```python
# Find all links with a specific class and href attribute
links = soup.select('a.external-link[href^="http"]')

# Get the first paragraph's text, stripped of leading/trailing whitespace
first_paragraph_text = soup.select_one('div#content p').get_text(strip=True)

# Remove an unwanted element from the tree
unwanted_ad = soup.select_one('.ad-banner')
if unwanted_ad:
    unwanted_ad.decompose() # Removes the tag and its contents
```

**❌ BAD: Over-reliance on `find_all` with complex dictionaries, raw text.**
```python
# Less readable and potentially less efficient
links = soup.find_all('a', class_='external-link', href=lambda h: h and h.startswith('http'))

# Text with extra newlines and spaces
first_paragraph_text = soup.find('div', id='content').find('p').text
```

## 5. Handling Dynamic Content

Beautiful Soup is for static HTML. If a page relies on JavaScript to render content, use `Selenium` or `Playwright` to render the page first, then pass the *rendered* HTML to Beautiful Soup. Do not try to make BS4 handle JavaScript.

**✅ GOOD: Use Selenium/Playwright for rendering, then BS4 for parsing.**
```python
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
#
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
# driver.get("https://js-rendered-site.com")
# rendered_html = driver.page_source
# driver.quit()
#
# soup = BeautifulSoup(rendered_html, 'lxml')
# # ... proceed with BS4 parsing ...
```

**❌ BAD: Expecting BS4 to execute JavaScript.**
```python
# This will only parse the initial static HTML, not JS-rendered content
# soup = BeautifulSoup(requests.get("https://js-rendered-site.com").text, 'lxml')
```