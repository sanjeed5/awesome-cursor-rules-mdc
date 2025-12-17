---
description: Definitive guidelines for using cheerio effectively in Node.js projects, focusing on robust web scraping, performance, and maintainable code.
globs: **/*.{js,jsx}
---
# cheerio Best Practices

`cheerio` is your lightweight, server-side jQuery for Node.js, ideal for fast HTML parsing and manipulation. Follow these rules to build efficient, maintainable, and robust scraping solutions.

## 1. Code Organization & Structure

**Always modularize your scraping logic.** Separate HTTP requests, HTML parsing, and data persistence. Use `async/await` for all asynchronous operations.

### ✅ GOOD: Modular & Asynchronous Structure

```javascript
// index.js
import * as cheerio from 'cheerio';
import axios from 'axios';
import { writeFile } from 'fs/promises';

async function fetchAndParse(url) {
  try {
    const { data: html } = await axios.get(url, {
      headers: { 'User-Agent': 'YourScraper/1.0' }
    });
    const $ = cheerio.load(html);

    const products = [];
    $('.product-item').each((i, el) => {
      const $el = $(el);
      const name = $el.find('.product-name').text().trim();
      const price = $el.find('.product-price').text().trim();
      const imageUrl = $el.find('img').attr('src');
      
      // Always check for existence before pushing
      if (name) { 
        products.push({ name, price, imageUrl });
      }
    });
    return products;
  } catch (error) {
    console.error(`Error scraping ${url}:`, error.message);
    throw error;
  }
}

async function main() {
  const targetUrl = 'https://www.example.com/products';
  const scrapedData = await fetchAndParse(targetUrl);
  await writeFile('products.json', JSON.stringify(scrapedData, null, 2), 'utf8');
  console.log('Scraping complete. Data saved to products.json');
}

main();
```

## 2. Common Patterns & Anti-patterns

### 2.1. Selector Efficiency

**Keep selectors specific and scoped.** Avoid broad selectors that force `cheerio` to traverse large portions of the DOM. Chain `.find()` for nested elements.

```javascript
// ❌ BAD: Overly broad, inefficient
const allLinks = $('a').map((i, el) => $(el).attr('href')).get();

// ✅ GOOD: Scoped to relevant section
const productLinks = $('#product-grid a.product-link').map((i, el) => $(el).attr('href')).get();

// ✅ GOOD: Chain .find() for precision
const productName = $('.product-card').first().find('h2.title').text().trim();
```

### 2.2. Data Extraction & Cleaning

**Use `.text()` for visible text, `.attr()` for attributes, `.html()` for inner HTML.** Always `.trim()` extracted text to remove whitespace.

```javascript
// ❌ BAD: Untrimmed text, can contain leading/trailing whitespace or newlines
const title = $('h1').text(); // "  My Product Title  \n"

// ✅ GOOD: Clean text extraction
const title = $('h1').text().trim(); // "My Product Title"

// ❌ BAD: Assuming element exists, `undefined.text()` will crash
const price = $('.product-price').text().trim(); 

// ✅ GOOD: Check element existence before extracting
const $priceEl = $('.product-price');
const price = $priceEl.length ? $priceEl.text().trim() : 'N/A';
```

## 3. Performance Considerations

### 3.1. Large HTML Documents (>10MB)

`cheerio.load()` creates an in-memory DOM. For very large documents, this can consume significant memory.

**Strategies for large documents:**
1.  **Selective Loading:** If possible, extract and parse only relevant HTML snippets.
2.  **Clear References:** Set `$` to `null` after processing to aid garbage collection.

```javascript
// ✅ GOOD: Aid garbage collection
let $ = cheerio.load(htmlContent);
// ... perform operations ...
$ = null; // Clear reference
if (global.gc) global.gc(); // Force GC (requires --expose-gc)
```

### 3.2. Concurrency & Rate Limiting

For scalable, production scraping, **use `Crawlee`'s `CheerioCrawler`**. It handles request queuing, concurrency, retries, and rate limiting automatically.

```javascript
// ✅ GOOD: Scalable scraping with Crawlee
import { CheerioCrawler, Dataset } from 'crawlee';

const crawler = new CheerioCrawler({
    maxRequestsPerMinute: 60, // Respect rate limits
    async requestHandler({ request, $, log }) {
        log.info(`Processing ${request.url}`);
        const title = $('h1').text().trim();
        await Dataset.pushData({ url: request.url, title });
    },
});

await crawler.run(['https://crawlee.dev']);
```

## 4. Common Pitfalls & Gotchas

### 4.1. Cheerio is NOT a Browser

**`cheerio` does not execute JavaScript, render CSS, or load external resources.** It only parses the raw HTML string.
**Solution:** For dynamic content rendered by client-side JavaScript, use browser automation tools like `Puppeteer` or `Playwright`.

### 4.2. Respect `robots.txt`

**Always check and respect a website's `robots.txt` file.** Ethical scraping is non-negotiable.

## 5. Testing Approaches

**Unit test your parsing logic with mocked HTML.** This ensures your selectors and extraction methods are robust and resilient to minor HTML changes.

```javascript
// parser.test.js (using Jest)
import * as cheerio from 'cheerio';

describe('Product Parser', () => {
  const mockHtml = `
    <div class="product-item">
      <img src="/img/prod1.jpg">
      <h2 class="product-name">Product A</h2>
      <span class="product-price">$19.99</span>
    </div>`;

  it('should extract product details correctly', () => {
    const $ = cheerio.load(mockHtml);
    const $product = $('.product-item');
    expect($product.find('.product-name').text().trim()).toBe('Product A');
    expect($product.find('.product-price').text().trim()).toBe('$19.99');
    expect($product.find('img').attr('src')).toBe('/img/prod1.jpg');
  });

  it('should handle missing elements gracefully', () => {
    const $ = cheerio.load('<div class="product-item"></div>');
    const $product = $('.product-item');
    expect($product.find('.non-existent').text().trim()).toBe(''); // Cheerio returns empty string
    expect($product.find('img').attr('src')).toBeUndefined(); // Cheerio returns undefined for missing attr
  });
});
```