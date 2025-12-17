---
description: This guide provides definitive, actionable best practices for writing modern D3.js v7+ code, focusing on modularity, performance, and maintainability.
globs: **/*.{js,jsx}
---
# d3 Best Practices

D3.js is a powerful, low-level library for data visualization. To leverage its full potential and maintain a clean, performant codebase, adhere to these definitive guidelines.

## 1. Code Organization and Structure

Always structure your D3 visualizations with clear separation of concerns. This enhances readability, testability, and reusability.

### 1.1. Modular Imports

Import only the specific D3 modules you need. This is critical for bundle size optimization and tree-shaking.

❌ **BAD: Over-importing**
```javascript
// Imports the entire D3 library, increasing bundle size unnecessarily.
import * as d3 from 'd3'; 

// Or using a CDN for the full library
// <script src="https://d3js.org/d3.v7.min.js"></script>
```

✅ **GOOD: Targeted Imports**
```javascript
// Import only necessary modules for a simple bar chart
import { select } from 'd3-selection';
import { scaleLinear, scaleBand } from 'd3-scale';
import { axisBottom, axisLeft } from 'd3-axis';
import { max } from 'd3-array';
```

### 1.2. Functional Encapsulation

Organize your visualization logic into distinct, focused functions. A single function should handle data loading, another for scale creation, another for axis rendering, and so on.

```javascript
// chart.js
import { select } from 'd3-selection';
import { scaleLinear, scaleBand } from 'd3-scale';
import { axisBottom, axisLeft } from 'd3-axis';
import { max } from 'd3-array';

export function createBarChart(containerSelector, data, options = {}) {
  const { width = 800, height = 500, margin = { top: 20, right: 20, bottom: 30, left: 40 } } = options;

  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const svg = select(containerSelector)
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`) // Responsive design
    .attr('role', 'img')
    .attr('aria-label', 'Bar chart showing data distribution');

  const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // 1. Create Scales
  const xScale = scaleBand()
    .domain(data.map(d => d.category))
    .range([0, innerWidth])
    .padding(0.1);

  const yScale = scaleLinear()
    .domain([0, max(data, d => d.value)])
    .range([innerHeight, 0]);

  // 2. Render Axes
  g.append('g')
    .attr('class', 'x-axis') // Style with CSS
    .attr('transform', `translate(0,${innerHeight})`)
    .call(axisBottom(xScale));

  g.append('g')
    .attr('class', 'y-axis') // Style with CSS
    .call(axisLeft(yScale));

  // 3. Draw Elements (Bars)
  g.selectAll('.bar')
    .data(data)
    .join('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.category))
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.value));

  // Add labels, title, etc. as separate concerns
}

// main.js
import { createBarChart } from './chart.js';

async function init() {
  const chartData = await fetch('/api/data').then(res => res.json());
  createBarChart('#chart-container', chartData);
}

init();
```

## 2. Common Patterns and Anti-patterns

Embrace D3's idiomatic patterns for data binding and DOM manipulation.

### 2.1. The `selection.join()` Pattern

Always use `selection.join()` for data-driven element creation, update, and removal. It is the most concise and efficient way to handle the enter, update, and exit selections.

❌ **BAD: Manual Enter/Update/Exit (D3 v3/v4 style)**
```javascript
const bars = g.selectAll('.bar')
  .data(data);

// Exit
bars.exit().remove();

// Enter
bars.enter().append('rect')
  .attr('class', 'bar')
  .attr('x', d => xScale(d.category))
  .attr('width', xScale.bandwidth())
  .attr('y', innerHeight) // Initial position for animation
  .attr('height', 0) // Initial height for animation
  .transition().duration(500) // Animate enter
    .attr('y', d => yScale(d.value))
    .attr('height', d => innerHeight - yScale(d.value));

// Update (merged selection)
bars.transition().duration(500)
  .attr('x', d => xScale(d.category))
  .attr('y', d => yScale(d.value))
  .attr('width', xScale.bandwidth())
  .attr('height', d => innerHeight - yScale(d.value));
```

✅ **GOOD: `selection.join()` (D3 v5+ style)**
```javascript
g.selectAll('.bar')
  .data(data, d => d.category) // Use a key function for stable joins
  .join(
    enter => enter.append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.category))
      .attr('y', innerHeight) // Start from bottom for animation
      .attr('width', xScale.bandwidth())
      .attr('height', 0) // Start with zero height for animation
      .call(enter => enter.transition().duration(500)
        .attr('y', d => yScale(d.value))
        .attr('height', d => innerHeight - yScale(d.value))),
    update => update.transition().duration(500)
      .attr('x', d => xScale(d.category))
      .attr('y', d => yScale(d.value))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.value)),
    exit => exit.transition().duration(500)
      .attr('y', innerHeight) // Animate to bottom
      .attr('height', 0) // Animate to zero height
      .remove()
  );
```

### 2.2. External CSS for Styling

Style D3 elements using external CSS classes. Avoid inline `style()` or `attr()` calls for visual properties like `fill`, `stroke`, `font-size`. This improves maintainability, themeability, and accessibility.

❌ **BAD: Inline Styling**
```javascript
g.append('g')
  .attr('transform', `translate(0,${innerHeight})`)
  .call(axisBottom(xScale))
  .selectAll('text')
    .attr('font-size', '12px')
    .attr('fill', '#333');
```

✅ **GOOD: CSS Classes**
```javascript
// In your JavaScript
g.append('g')
  .attr('class', 'x-axis') // Apply a class
  .attr('transform', `translate(0,${innerHeight})`)
  .call(axisBottom(xScale));

// In your CSS file (e.g., style.css)
.x-axis text {
  font-size: 12px;
  fill: #333;
}
.x-axis path, .x-axis line {
  stroke: currentColor; /* Inherits color from parent */
}
```

### 2.3. D3 Behaviors for Interaction

For common interactions like zooming, panning, and dragging, use D3's dedicated behavior modules (`d3-zoom`, `d3-drag`, `d3-brush`). They handle complex event logic and browser quirks robustly.

❌ **BAD: Manual Zoom/Pan Implementation**
```javascript
// Don't try to reimplement zoom/pan with raw mouse events.
svg.on('mousedown', handleMouseDown);
svg.on('mousemove', handleMouseMove);
// ... this path leads to bugs and frustration.
```

✅ **GOOD: Using `d3-zoom`**
```javascript
import { zoom, zoomIdentity } from 'd3-zoom';
import { select } from 'd3-selection';

function zoomed({ transform }) {
  // Apply the transform to your chart elements
  g.attr('transform', transform);
  // Update axes if necessary
  xAxisGroup.call(axisBottom(transform.rescaleX(xScale)));
  yAxisGroup.call(axisLeft(transform.rescaleY(yScale)));
}

const chartZoom = zoom()
  .scaleExtent([0.5, 10]) // Min/max zoom level
  .translateExtent([[0, 0], [width, height]]) // Pan boundaries
  .on('zoom', zoomed);

select('#chart-container svg')
  .call(chartZoom)
  .call(chartZoom.transform, zoomIdentity); // Initialize zoom
```

## 3. Performance Considerations

Optimize D3 charts for smooth interactions and efficient rendering, especially with large datasets.

### 3.1. Use `viewBox` for Responsiveness

Always use the `viewBox` attribute on your SVG element for responsive scaling. Avoid manually recalculating and setting `width`/`height` attributes on resize events.

❌ **BAD: Fixed Dimensions, Manual Resize Handling**
```javascript
const svg = d3.select('#chart').append('svg')
  .attr('width', 960)
  .attr('height', 500);

// Requires complex resize observer logic to update width/height and redraw
window.addEventListener('resize', () => { /* ... */ });
```

✅ **GOOD: `viewBox` and CSS**
```javascript
// In your JavaScript
const svg = d3.select('#chart').append('svg')
  .attr('viewBox', `0 0 ${width} ${height}`); // Logical dimensions

// In your CSS
#chart svg {
  display: block; /* Remove extra space below SVG */
  width: 100%;    /* Scale to container width */
  height: auto;   /* Maintain aspect ratio */
}
```

### 3.2. Minimize DOM Operations

Batch DOM updates and avoid unnecessary re-renders. `selection.join()` inherently helps with this by efficiently managing enter/update/exit.

### 3.3. Consider Canvas for Large Datasets

For extremely large datasets (thousands or millions of elements), SVG can become slow. D3 can draw to HTML Canvas elements, which offer superior performance for high-volume rendering.

```javascript
// Example: Drawing a scatter plot on Canvas
import { select } from 'd3-selection';
// ... other D3 modules

const canvas = select('#chart-container')
  .append('canvas')
  .attr('width', width)
  .attr('height', height)
  .node();

const context = canvas.getContext('2d');

function drawCanvasChart(data) {
  context.clearRect(0, 0, width, height);
  context.fillStyle = 'steelblue';
  data.forEach(d => {
    context.beginPath();
    context.arc(xScale(d.x), yScale(d.y), 3, 0, 2 * Math.PI);
    context.fill();
  });
}

// Call drawCanvasChart(data) whenever data or scales change.
```

## 4. Common Pitfalls and Gotchas

Avoid these common mistakes to prevent bugs and maintain a robust D3 application.

### 4.1. Forgetting Key Functions with `data()`

When updating data, always provide a key function to `selection.data(newData, d => d.id)`. This tells D3 how to uniquely identify data points, ensuring stable joins and correct element transitions.

❌ **BAD: No Key Function (unstable joins)**
```javascript
// If data order changes or items are added/removed, D3 might re-bind incorrectly.
g.selectAll('.bar').data(data)
  .join(/* ... */);
```

✅ **GOOD: With Key Function**
```javascript
// D3 uses d.category to match old and new data points.
g.selectAll('.bar').data(data, d => d.category)
  .join(/* ... */);
```

### 4.2. Incorrectly Updating Scales and Axes

When data or dimensions change, you must update the scales' domains/ranges *before* calling the axis generators again.

❌ **BAD: Updating Axes without updating Scales**
```javascript
// This will render axes based on old scale domains/ranges.
xAxisGroup.call(axisBottom(xScale)); // xScale's domain/range might be stale
```

✅ **GOOD: Update Scales, then Axes**
```javascript
xScale.domain(data.map(d => d.category));
yScale.domain([0, max(data, d => d.value)]);

xAxisGroup.transition().duration(500).call(axisBottom(xScale));
yAxisGroup.transition().duration(500).call(axisLeft(yScale));
```

### 4.3. Ignoring Accessibility (ARIA Attributes)

D3 charts are often visual. For screen readers and other assistive technologies, add semantic SVG elements and ARIA attributes.

```javascript
const svg = select(containerSelector)
  .append('svg')
  .attr('role', 'img') // Indicate it's an image
  .attr('aria-label', 'Bar chart showing monthly sales data'); // Descriptive label

// For individual elements
g.selectAll('.bar')
  .data(data, d => d.category)
  .join('rect')
    .attr('aria-label', d => `Sales for ${d.category}: ${d.value}`);
```

## 5. Testing Approaches

Ensure the reliability and correctness of your D3 visualizations.

### 5.1. Unit Testing Pure Functions

Test your data transformation, scale creation, and utility functions independently. These are typically pure functions, making them easy to test.

```javascript
// data-utils.js
export function processRawData(rawData) {
  return rawData.map(d => ({
    category: d.name,
    value: +d.amount // Ensure numeric type
  }));
}

// data-utils.test.js (using Jest)
import { processRawData } from './data-utils';

test('processRawData converts amount to number', () => {
  const raw = [{ name: 'A', amount: '100' }];
  const processed = processRawData(raw);
  expect(processed[0].value).toBe(100);
});
```

### 5.2. Visual Regression Testing

For visual output, implement visual regression tests using tools like Playwright or Cypress. These tools capture screenshots of your charts and compare them against baseline images, flagging any unintended visual changes.

```javascript
// cypress/e2e/chart.cy.js
describe('Bar Chart Visuals', () => {
  it('should render correctly', () => {
    cy.visit('/charts/bar-chart'); // Your chart page
    cy.get('#chart-container svg').compareSnapshot('bar-chart-initial');
  });

  it('should update correctly on data change', () => {
    cy.visit('/charts/bar-chart?data=updated'); // Page with updated data
    cy.get('#chart-container svg').compareSnapshot('bar-chart-updated');
  });
});
```

### 5.3. Observable Notebooks for Prototyping

Use Observable notebooks for rapid prototyping, sharing, and testing D3 code snippets. They provide an interactive environment to validate visual output and explore D3 patterns.