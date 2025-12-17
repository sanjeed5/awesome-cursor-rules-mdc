---
description: This guide provides opinionated, actionable best practices for writing robust, performant, and maintainable Puppeteer scripts for web scraping, automation, and testing. It emphasizes modern patterns like `page.locator()` and efficient resource management.
globs: **/*.{js,jsx}
---
# puppeteer Best Practices

Puppeteer is the definitive library for headless browser automation in Node.js. Adhere to these guidelines to build reliable, fast, and maintainable scripts.

## 1. Foundation: Async/Await & Browser Management

Always use `async/await` for clear, sequential asynchronous code. Manage browser instances carefully.

```javascript
// ❌ BAD: Synchronous or callback-hell
// puppeteer.launch().then(browser => { ... });

// ✅ GOOD: Clear async/await structure
import puppeteer from 'puppeteer';

async function runAutomation() {
  const browser = await puppeteer.launch({ headless: 'new' }); // Always use 'new' headless mode
  const page = await browser.newPage();

  try {
    // ... automation logic ...
  } catch (error) {
    console.error('Automation failed:', error);
  } finally {
    await browser.close(); // Ensure browser always closes
  }
}

runAutomation();
```

## 2. Interactions: Embrace `page.locator()`

`page.locator()` is the modern, robust way to interact with elements. It automatically handles waiting for elements to be visible, enabled, and stable, eliminating race conditions.

```javascript
// ❌ BAD: Manual waiting and brittle selectors
// await page.waitForSelector('.my-button');
// await page.click('.my-button');
// await page.$eval('h1', el => el.innerText); // No auto-waiting for element presence

// ✅ GOOD: Use page.locator() for all interactions
async function interactWithPage(page) {
  // Click a button
  await page.locator('button.submit-btn').click();

  // Fill an input field
  await page.locator('input[name="username"]').fill('myUsername');

  // Extract text with auto-waiting
  const title = await page.locator('h1.page-title').evaluate(el => el.innerText);
  console.log('Page Title:', title);

  // Wait for an element to appear without interacting
  await page.locator('.loading-spinner').wait({ state: 'hidden' }); // Wait for it to disappear
}
```

## 3. Efficiency & Performance

Optimize Puppeteer scripts by reducing unnecessary resource loading and leveraging parallel execution.

### 3.1. Disable Unnecessary Resources

Block images, CSS, and fonts to significantly speed up page loads, especially for scraping.

```javascript
// ❌ BAD: Loading all resources, wasting bandwidth and time
// await page.goto('https://example.com');

// ✅ GOOD: Intercept and block non-essential resources
async function optimizePageLoad(page) {
  await page.setRequestInterception(true);
  page.on('request', (request) => {
    if (['image', 'stylesheet', 'font', 'media'].includes(request.resourceType())) {
      request.abort();
    } else {
      request.continue();
    }
  });
  await page.goto('https://example.com', { waitUntil: 'domcontentloaded' }); // Use 'domcontentloaded' when possible
}
```

### 3.2. Parallelize Operations with `Promise.all()`

When processing multiple pages or independent tasks, run them concurrently.

```javascript
// ❌ BAD: Sequential processing of multiple URLs
// for (const url of urls) { await processUrl(page, url); }

// ✅ GOOD: Process multiple URLs in parallel
async function processMultipleUrls(browser, urls) {
  const pagePromises = urls.map(async (url) => {
    const page = await browser.newPage();
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded' });
      const data = await page.locator('h1').evaluate(el => el.innerText);
      return { url, data };
    } catch (error) {
      console.error(`Failed to process ${url}:`, error);
      return { url, error: error.message };
    } finally {
      await page.close(); // Close page after processing
    }
  });
  return Promise.all(pagePromises);
}
```

## 4. Robustness & Anti-Bot Measures

Build resilient scrapers that handle failures gracefully and avoid detection.

### 4.1. Implement Retry Logic with Exponential Back-off

Network issues and transient errors are common. Retry failed operations with increasing delays.

```javascript
// ❌ BAD: No retry logic, script fails on first error
// await page.click('.flaky-button');

// ✅ GOOD: Robust retry mechanism
async function clickWithRetry(page, selector, retries = 3, delay = 1000) {
  for (let i = 0; i < retries; i++) {
    try {
      await page.locator(selector).click({ timeout: 5000 });
      return; // Success
    } catch (error) {
      console.warn(`Attempt ${i + 1} failed for ${selector}: ${error.message}`);
      if (i < retries - 1) {
        await page.waitForTimeout(delay * Math.pow(2, i)); // Exponential back-off
      } else {
        throw new Error(`Failed to click ${selector} after ${retries} attempts.`);
      }
    }
  }
}
// Usage: await clickWithRetry(page, 'button.flaky-submit');
```

### 4.2. Integrate Stealth Plugin for Anti-Bot Evasion

Websites actively detect automated browsers. Use `puppeteer-extra` and `puppeteer-extra-plugin-stealth`.

```javascript
// ❌ BAD: Easily detectable as a bot
// const browser = await puppeteer.launch();

// ✅ GOOD: Launch with stealth plugin
import puppeteer from 'puppeteer-extra';
import StealthPlugin from 'puppeteer-extra-plugin-stealth';

puppeteer.use(StealthPlugin());

async function launchStealthBrowser() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox', // Essential for CI/Docker
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage', // Essential for CI/Docker
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--single-process', // For CI/Docker
      '--disable-gpu'
    ]
  });
  const page = await browser.newPage();
  await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
  await page.setViewport({ width: 1920, height: 1080 }); // Realistic viewport
  return { browser, page };
}
```

## 5. Code Quality: Modularization & Typing

Organize your code into focused, reusable functions. Consider TypeScript for larger projects.

```javascript
// ❌ BAD: Monolithic script
// (async () => { /* hundreds of lines of mixed logic */ })();

// ✅ GOOD: Modular functions for clarity and reusability
// scraper.js
import puppeteer from 'puppeteer';

async function navigateAndLogin(page, username, password) {
  await page.goto('https://example.com/login');
  await page.locator('#username').fill(username);
  await page.locator('#password').fill(password);
  await page.locator('button[type="submit"]').click();
  await page.waitForNavigation({ waitUntil: 'networkidle0' });
}

async function extractProductData(page) {
  const products = await page.locator('.product-item').evaluateAll(items =>
    items.map(item => ({
      name: item.querySelector('.product-name')?.innerText,
      price: item.querySelector('.product-price')?.innerText,
    }))
  );
  return products;
}

export async function scrapeWebsite(url, credentials) {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  try {
    await navigateAndLogin(page, credentials.username, credentials.password);
    await page.goto(url);
    const data = await extractProductData(page);
    return data;
  } finally {
    await browser.close();
  }
}

// main.js
// import { scrapeWebsite } from './scraper.js';
// (async () => {
//   const data = await scrapeWebsite('https://example.com/products', { username: 'user', password: 'pass' });
//   console.log(data);
// })();
```