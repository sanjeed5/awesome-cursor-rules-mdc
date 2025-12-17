---
description: This guide provides opinionated, actionable best practices for building robust, high-performance, and maintainable Scrapy web crawlers, emphasizing modern Python standards and ethical scraping.
globs: **/*.py
---
# Scrapy Best Practices

Scrapy is the definitive framework for high-performance web scraping in Python. This guide outlines the essential practices for building resilient, scalable, and ethical crawlers. Adhere to these guidelines to ensure your Scrapy projects are maintainable, efficient, and robust against the dynamic web landscape of 2025.

## 1. Code Organization and Structure

Maintain a clean, logical project structure. This enhances readability, testability, and scalability.

### 1.1. Standard Project Layout

Always use `scrapy startproject` to initialize your project. This sets up the recommended directory structure.

**❌ BAD: Manually creating files and directories**
```
# Don't do this
mkdir my_scraper
cd my_scraper
touch scrapy.cfg items.py spiders/__init__.py ...
```

**✅ GOOD: Use the Scrapy CLI**
```bash
scrapy startproject my_project_name
cd my_project_name
```

### 1.2. Item Definitions (`items.py`)

Define your data models clearly using `scrapy.Item` subclasses. Always include type hints for better IDE support and code clarity.

**❌ BAD: Generic dictionaries or untyped `Item` fields**
```python
# items.py
import scrapy

class ProductItem(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    # No type hints, hard to know expected data type
```

**✅ GOOD: `scrapy.Item` with `Field` and `typing` hints**
```python
# items.py
import scrapy
from scrapy.item import Field
from typing import Optional, List

class ProductItem(scrapy.Item):
    url: str = Field()
    title: Optional[str] = Field()
    price: Optional[float] = Field()
    description: Optional[str] = Field()
    image_urls: List[str] = Field()
    category: Optional[str] = Field()
```

### 1.3. Spiders

Keep spiders focused on crawling logic and initial data extraction. They should yield `Request` objects and `Item` objects.

**❌ BAD: Complex data processing or storage logic in spiders**
```python
# spiders/bad_spider.py
import scrapy
from my_project_name.items import ProductItem

class BadSpider(scrapy.Spider):
    name = "bad_spider"
    start_urls = ["http://example.com"]

    def parse(self, response):
        item = ProductItem()
        item['title'] = response.css('h1::text').get()
        # ... complex cleaning and validation here ...
        # ... database insertion logic here ...
        yield item
```

**✅ GOOD: Spiders for crawling and raw extraction, pipelines for processing**
```python
# spiders/good_spider.py
import scrapy
from my_project_name.items import ProductItem

class GoodSpider(scrapy.Spider):
    name = "good_spider"
    allowed_domains = ["example.com"] # Always define allowed_domains
    start_urls = ["http://example.com/products"]

    def parse(self, response: scrapy.http.Response):
        # Extract product links and follow them
        for product_link in response.css('a.product-link::attr(href)').getall():
            yield response.follow(product_link, callback=self.parse_product)

        # Handle pagination
        next_page = response.css('a.next-page::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_product(self, response: scrapy.http.Response):
        item = ProductItem(
            url=response.url,
            title=response.css('h1.product-title::text').get(),
            price=float(response.css('.product-price::text').get().replace('$', '')) if response.css('.product-price::text').get() else None,
            description=response.css('.product-description::text').get(),
            image_urls=response.css('.product-image::attr(src)').getall(),
            category=response.css('.product-category::text').get(),
        )
        yield item # Item is yielded for pipelines to process
```

### 1.4. Item Pipelines (`pipelines.py`)

Use pipelines for all data cleaning, validation, deduplication, and storage. This isolates business logic and makes testing straightforward.

**❌ BAD: No pipelines, processing in spiders or external scripts**
```python
# (See BAD spider example above)
# Or processing data after the crawl finishes in a separate script
```

**✅ GOOD: Dedicated pipelines for each processing step**
```python
# pipelines.py
from itemadapter import ItemAdapter
from my_project_name.items import ProductItem
from scrapy.exceptions import DropItem
from typing import Any

class PriceConverterPipeline:
    def process_item(self, item: ProductItem, spider: Any) -> ProductItem:
        adapter = ItemAdapter(item)
        if adapter.get('price'):
            try:
                # Ensure price is a float
                adapter['price'] = float(adapter['price'])
            except (ValueError, TypeError):
                raise DropItem(f"Invalid price in {item}")
        return item

class DuplicatesPipeline:
    def __init__(self):
        self.urls_seen = set()

    def process_item(self, item: ProductItem, spider: Any) -> ProductItem:
        adapter = ItemAdapter(item)
        if adapter['url'] in self.urls_seen:
            raise DropItem(f"Duplicate item found: {item}")
        else:
            self.urls_seen.add(adapter['url'])
            return item

# In settings.py, enable pipelines and set order:
# ITEM_PIPELINES = {
#     "my_project_name.pipelines.PriceConverterPipeline": 300,
#     "my_project_name.pipelines.DuplicatesPipeline": 400,
#     # ... other pipelines for storage (e.g., to database)
# }
```

### 1.5. Middlewares (`middlewares.py`)

Implement custom downloader and spider middlewares for request/response manipulation, proxy rotation, user-agent management, and advanced error handling.

**❌ BAD: Hardcoding user-agents or proxies in spiders**
```python
# spiders/bad_spider.py
# ...
def start_requests(self):
    yield scrapy.Request(
        url="http://example.com",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        meta={"proxy": "http://my.proxy.com:8080"}
    )
```

**✅ GOOD: Centralized middleware for robust request handling**
```python
# middlewares.py
import random
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.exceptions import IgnoreRequest
from typing import Any

class RotateUserAgentMiddleware:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        # ... add more diverse user agents
    ]

    def process_request(self, request: Request, spider: Any) -> None:
        request.headers['User-Agent'] = random.choice(self.USER_AGENTS)

class ProxyMiddleware:
    PROXIES = [
        "http://user:pass@proxy1.com:8080",
        "http://user:pass@proxy2.com:8080",
        # ... add more proxies
    ]

    def process_request(self, request: Request, spider: Any) -> None:
        if not request.meta.get('proxy'): # Only apply if not already set
            request.meta['proxy'] = random.choice(self.PROXIES)

# In settings.py, enable middlewares and set order:
# DOWNLOADER_MIDDLEWARES = {
#     "my_project_name.middlewares.RotateUserAgentMiddleware": 543,
#     "my_project_name.middlewares.ProxyMiddleware": 544,
# }
```

## 2. Common Patterns and Anti-patterns

### 2.1. Running Spiders from a Script

Always use `scrapy.crawler.CrawlerProcess` for launching spiders from a Python script. It handles the Twisted reactor and project settings correctly.

**❌ BAD: Manually managing the Twisted reactor or `CrawlerRunner` for simple cases**
```python
# run_bad.py
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor
# ... spider definition ...
runner = CrawlerRunner()
d = runner.crawl(MySpider)
d.addBoth(lambda _: reactor.stop())
reactor.run() # Too verbose for common use
```

**✅ GOOD: `scrapy.crawler.CrawlerProcess`**
```python
# run_good.py
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from typing import Any

class MySpider(scrapy.Spider):
    name = "my_spider"
    start_urls = ["http://quotes.toscrape.com/"]

    def parse(self, response: scrapy.http.Response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
            }

if __name__ == '__main__':
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    process.crawl(MySpider)
    process.start() # The script will block here until the crawling is finished
```

### 2.2. Robust Request Handling

Enable `AUTOTHROTTLE` and set `DOWNLOAD_DELAY`. Implement exponential backoff for 403/429 responses via a custom middleware or by configuring `RETRY_HTTP_CODES`.

**❌ BAD: Ignoring rate limits, getting banned**
```python
# settings.py
# No AUTOTHROTTLE, no DOWNLOAD_DELAY
# RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408] # Default, misses 403/429
```

**✅ GOOD: Respectful crawling with `AUTOTHROTTLE` and extended `RETRY_HTTP_CODES`**
```python
# settings.py
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0 # Initial delay
AUTOTHROTTLE_MAX_DELAY = 60.0 # Max delay between requests
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0 # Aim for 1 request per second
AUTOTHROTTLE_DEBUG = False # Set to True for debugging

DOWNLOAD_DELAY = 0.5 # Minimum delay between requests to the same domain

# Enable retries for common anti-bot responses
RETRY_ENABLED = True
RETRY_TIMES = 5 # Retry up to 5 times
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 403, 429] # Add 403, 429
```

## 3. Performance Considerations

### 3.1. Concurrency

Tune `CONCURRENT_REQUESTS` and `CONCURRENT_REQUESTS_PER_DOMAIN` carefully. Start low and increase gradually while monitoring server load and your IP's ban rate.

**❌ BAD: Default high concurrency or excessively low concurrency**
```python
# settings.py
# CONCURRENT_REQUESTS = 16 # Default is often too high for sensitive sites
# DOWNLOAD_DELAY = 0 # No delay, will get banned quickly
```

**✅ GOOD: Balanced concurrency with `AUTOTHROTTLE`**
```python
# settings.py
CONCURRENT_REQUESTS = 32 # Maximum concurrent requests overall
CONCURRENT_REQUESTS_PER_DOMAIN = 8 # Maximum concurrent requests per domain
CONCURRENT_REQUESTS_PER_IP = 0 # Use CONCURRENT_REQUESTS_PER_DOMAIN instead
# AUTOTHROTTLE should be enabled to dynamically adjust delay
```

### 3.2. Distributed Crawling

For large-scale projects, leverage Scrapy Cluster or Scrapyd. Store crawl state (e.g., visited URLs, pending requests) in a persistent backend like Redis to enable resume after failures and distributed processing.

**✅ GOOD: Use `scrapy-redis` or similar for distributed state management**
```python
# settings.py (for scrapy-redis)
# DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"
# SCHEDULER = "scrapy_redis.scheduler.Scheduler"
# SCHEDULER_PERSIST = True
# REDIS_URL = 'redis://localhost:6379'
```

## 4. Common Pitfalls and Gotchas

### 4.1. Ignoring `robots.txt`

Always check and respect `robots.txt`. It's an ethical and practical necessity to avoid legal issues and IP bans.

**❌ BAD: Disabling `ROBOTSTXT_OBEY` without justification**
```python
# settings.py
ROBOTSTXT_OBEY = False # Don't do this unless absolutely necessary and legally cleared
```

**✅ GOOD: Always obey `robots.txt` by default**
```python
# settings.py
ROBOTSTXT_OBEY = True
```

### 4.2. Memory Leaks

Avoid storing large amounts of data in spider attributes. Yield items and requests promptly.

**❌ BAD: Appending all items to a list in the spider**
```python
# spiders/bad_spider.py
class BadSpider(scrapy.Spider):
    # ...
    all_items = [] # This will grow indefinitely and cause memory issues

    def parse(self, response):
        # ... extract item ...
        self.all_items.append(item)
        # ...
```

**✅ GOOD: Yield items and requests immediately**
```python
# spiders/good_spider.py
class GoodSpider(scrapy.Spider):
    # ...
    def parse(self, response):
        # ... extract item ...
        yield item # Item is sent to pipelines immediately
        # ... yield requests for next pages ...
        yield scrapy.Request(url="next_page", callback=self.parse)
```

## 5. Type Hints

Embrace type hints (`typing` module) throughout your Scrapy project. This improves code quality, enables static analysis, and makes your code easier to understand and maintain.

**❌ BAD: Untyped functions and variables**
```python
# pipelines.py
class MyPipeline:
    def process_item(self, item, spider):
        # ... logic ...
        return item
```

**✅ GOOD: Fully typed functions and variables**
```python
# pipelines.py
from itemadapter import ItemAdapter
from my_project_name.items import ProductItem
from typing import Any

class MyPipeline:
    def process_item(self, item: ProductItem, spider: Any) -> ProductItem:
        adapter = ItemAdapter(item)
        # ... logic ...
        return item
```

## 6. Virtual Environments

Always use a virtual environment (`venv` or `conda`) for Scrapy projects. This isolates dependencies and prevents conflicts.

**❌ BAD: Installing packages globally**
```bash
pip install scrapy # Installs into global Python environment
```

**✅ GOOD: Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install scrapy
```

## 7. Packaging

For larger projects, package your Scrapy application using `setuptools` and `pyproject.toml`. This makes deployment and dependency management robust.

**✅ GOOD: Use `pyproject.toml` for project metadata and dependencies**
```toml
# pyproject.toml
[project]
name = "my-scrapy-project"
version = "0.1.0"
description = "A Scrapy project for scraping X"
dependencies = [
    "scrapy>=2.13.0",
    "itemadapter>=0.8.0",
    # ... other dependencies
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

## 8. Testing Approaches

Implement a comprehensive testing strategy to ensure your crawlers are reliable and resilient to website changes.

### 8.1. Unit Testing Spiders

Write `pytest` suites that mock `scrapy.http.Response` objects. Test your parsing logic in isolation.

**❌ BAD: No tests, relying on manual verification**
```python
# No test files
```

**✅ GOOD: `pytest` with mocked responses**
```python
# tests/test_my_spider.py
import pytest
from scrapy.http import Response, Request
from scrapy.item import Item
from my_project_name.spiders.good_spider import GoodSpider

@pytest.fixture
def mock_response_product_list():
    # Simulate a product listing page
    html = """
    <html><body>
        <a class="product-link" href="/product/1">Product 1</a>
        <a class="product-link" href="/product/2">Product 2</a>
        <a class="next-page" href="/products?page=2">Next</a>
    </body></html>
    """
    return Response(url="http://example.com/products", body=html.encode('utf-8'))

@pytest.fixture
def mock_response_product_detail():
    # Simulate a product detail page
    html = """
    <html><body>
        <h1 class="product-title">Awesome Widget</h1>
        <span class="product-price">$99.99</span>
        <p class="product-description">A great widget.</p>
        <img class="product-image" src="/img/widget.jpg">
        <span class="product-category">Electronics</span>
    </body></html>
    """
    return Response(url="http://example.com/product/1", body=html.encode('utf-8'))

def test_parse_product_list(mock_response_product_list):
    spider = GoodSpider()
    results = list(spider.parse(mock_response_product_list))

    # Check that requests for product details are yielded
    assert any(isinstance(r, Request) and r.url == "http://example.com/product/1" for r in results)
    assert any(isinstance(r, Request) and r.url == "http://example.com/product/2" for r in results)
    # Check that a request for the next page is yielded
    assert any(isinstance(r, Request) and r.url == "http://example.com/products?page=2" for r in results)

def test_parse_product_detail(mock_response_product_detail):
    spider = GoodSpider()
    results = list(spider.parse_product(mock_response_product_detail))

    # Check that an Item is yielded with correct data
    assert len(results) == 1
    item = results[0]
    assert isinstance(item, Item)
    assert item['title'] == "Awesome Widget"
    assert item['price'] == 99.99
    assert item['url'] == "http://example.com/product/1"
```

### 8.2. Linting and Type Checking

Integrate `flake8`, `black`, and `mypy` into your CI/CD pipeline. Enforce code style and type correctness.

**✅ GOOD: Automated code quality checks**
```bash
# In your CI/CD pipeline or pre-commit hooks
black . --check
flake8 .
mypy .
```

## 9. Ethical Scraping

Always prioritize ethical scraping practices. This protects your project from legal issues and IP bans.

**✅ GOOD: Ethical scraping checklist**
*   **Obey `robots.txt`**: Always.
*   **Clear User-Agent**: Identify your crawler with contact info.
*   **Rate Limiting**: Use `AUTOTHROTTLE` and `DOWNLOAD_DELAY`.
*   **Handle Errors Gracefully**: Implement retries and backoff.
*   **Distributed Crawling**: Use proxies and rotate IPs.
*   **Avoid Overloading Servers**: Monitor your crawl speed and resource usage.