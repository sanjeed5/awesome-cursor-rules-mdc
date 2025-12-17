---
description: Definitive guidelines for writing performant, maintainable, and modern jQuery code, emphasizing best practices for DOM manipulation, event handling, and compatibility with jQuery 4.0 and ES6+.
globs: **/*.{js,jsx}
---
# jQuery Best Practices

jQuery remains a powerful tool for DOM manipulation, but its use in 2025 demands a performance-first mindset and adherence to modern JavaScript standards. This guide outlines the definitive best practices for writing fast, clean, and maintainable jQuery code.

## 1. Code Organization and Modern JavaScript

Always encapsulate your jQuery code within an Immediately Invoked Function Expression (IIFE) to prevent global scope pollution and ensure `$` correctly maps to `jQuery`, especially in environments like WordPress where `jQuery.noConflict()` is active. Embrace modern ES6+ syntax where possible.

### 1.1 Scope and `$.noConflict()`

**Always wrap your jQuery code in an IIFE.** This ensures `$` refers to `jQuery` and protects your code from global scope conflicts.

❌ BAD: Global scope pollution, `$` conflicts
```javascript
// Potentially conflicts with other libraries or scripts
$('.my-element').on('click', function() {
  console.log('Clicked!');
});
```

✅ GOOD: Isolated scope, `$` aliased correctly
```javascript
(function($) {
  // Your jQuery code here, $ is safely aliased to jQuery
  $('.my-element').on('click', function() {
    console.log('Clicked!');
  });
})(jQuery); // Pass jQuery object into the IIFE
```

### 1.2 Modern JavaScript Syntax

Integrate `const`, `let`, arrow functions, and template literals. jQuery code coexists perfectly with modern JS.

❌ BAD: Old-school `var` and `function` syntax
```javascript
var myElement = $('#old-element');
myElement.click(function() {
  var message = 'Element ' + myElement.attr('id') + ' clicked.';
  console.log(message);
});
```

✅ GOOD: Modern ES6+ syntax
```javascript
const myElement = $('#modern-element');
myElement.on('click', () => {
  const message = `Element ${myElement.attr('id')} clicked.`;
  console.log(message);
});
```

## 2. Performance Considerations

Prioritize performance by minimizing DOM interactions, optimizing selectors, and caching frequently used elements and values.

### 2.1 Cache Selectors

**Never re-query the DOM for the same element.** Store jQuery objects in variables for reuse. Prefix jQuery variables with `$` for clarity.

❌ BAD: Repeated DOM queries
```javascript
$('.item').addClass('active');
$('.item').css('background-color', 'blue');
$('.item').on('click', function() { /* ... */ });
```

✅ GOOD: Cache selectors for efficiency
```javascript
const $items = $('.item');
$items.addClass('active');
$items.css('background-color', 'blue');
$items.on('click', function() { /* ... */ });
```

### 2.2 Minimize DOM Updates

Batch DOM modifications. Build HTML fragments off-document or use `detach()` before making multiple changes, then re-attach once. Each DOM write triggers potential reflows.

❌ BAD: Repeated DOM appends in a loop
```javascript
const data = ['one', 'two', 'three'];
const $list = $('#my-list');
data.forEach(item => {
  $list.append(`<li>${item}</li>`); // Each append causes a DOM update
});
```

✅ GOOD: Batch DOM updates
```javascript
const data = ['one', 'two', 'three'];
const $list = $('#my-list');
const $fragment = $(document.createDocumentFragment()); // Create a document fragment
data.forEach(item => {
  $fragment.append(`<li>${item}</li>`);
});
$list.append($fragment); // Append once
```

### 2.3 Optimize Selectors

Use the most specific and efficient selectors possible. Prefer ID and class selectors over complex descendant or attribute selectors.

❌ BAD: Inefficient descendant selector
```javascript
// Sizzle has to traverse all divs, then check for 'my-class'
$('div .my-class').css('color', 'red');
```

✅ GOOD: Direct class selector
```javascript
// Direct and fast
$('.my-class').css('color', 'red');
```

### 2.4 Cache Loop Lengths

When iterating over jQuery collections or arrays, cache the length to avoid re-evaluating it on each iteration.

❌ BAD: Re-evaluating length in each loop iteration
```javascript
const $elements = $('.my-elements');
for (let i = 0; i < $elements.length; i++) {
  $elements.eq(i).addClass('processed');
}
```

✅ GOOD: Cache loop length
```javascript
const $elements = $('.my-elements');
const numElements = $elements.length; // Cache length
for (let i = 0; i < numElements; i++) {
  $elements.eq(i).addClass('processed');
}
```

## 3. Common Patterns and Anti-patterns

Adopt patterns that promote maintainability and performance.

### 3.1 Delegated Events

**Always use event delegation for dynamic content or large collections of elements.** Attach a single event handler to a static parent element (or `document`) and use the selector argument of `.on()`.

❌ BAD: Attaching many individual event handlers
```javascript
// Attaches a handler to *every* current button. New buttons won't work.
$('button.delete').on('click', function() {
  $(this).closest('.item').remove();
});
```

✅ GOOD: Event delegation
```javascript
// Attaches one handler to the document, efficient for dynamic content
$(document).on('click', 'button.delete', function() {
  $(this).closest('.item').remove();
});
```

### 3.2 Element-Specific State with `$.data()`

Use `$.data()` to store and retrieve arbitrary data associated with DOM elements. This is cleaner and more memory-safe than attaching custom properties directly to DOM nodes.

❌ BAD: Attaching custom properties directly
```javascript
const $el = $('#my-element');
$el[0].customState = { count: 0 }; // Pollutes DOM node
$el.on('click', function() {
  $el[0].customState.count++;
  console.log($el[0].customState.count);
});
```

✅ GOOD: Using `$.data()`
```javascript
const $el = $('#my-element');
$el.data('customState', { count: 0 }); // Cleanly stores data
$el.on('click', function() {
  const state = $el.data('customState');
  state.count++;
  $el.data('customState', state); // Update data
  console.log($el.data('customState').count);
});
```

## 4. Common Pitfalls and Gotchas

Be aware of jQuery's nuances and recent changes, especially with jQuery 4.0.

### 4.1 `$(document).ready()` Syntax

As of jQuery 3.0, only `$(handler)` is the recommended syntax for DOM ready. Avoid deprecated forms.

❌ BAD: Deprecated or misleading `ready` syntaxes
```javascript
$(document).ready(function() { /* ... */ }); // Deprecated
$('img').ready(function() { /* ... */ });     // Misleading, doesn't wait for images
$(document).on('ready', function() { /* ... */ }); // Removed in jQuery 3.0
```

✅ GOOD: Recommended `ready` syntax
```javascript
$(function() {
  // Code executes when the DOM is fully loaded.
  console.log('DOM is ready!');
});
```

### 4.2 jQuery 4.0 API Changes

Stay updated with the jQuery 4.0 upgrade guide. Key breaking changes include `toggleClass(Boolean|undefined)` removal and updated Ajax API.

❌ BAD: Using `toggleClass(Boolean)` (removed in 4.0)
```javascript
// This will break in jQuery 4.0
$('.my-element').toggleClass('active', true);
```

✅ GOOD: Use `addClass()` or `removeClass()` explicitly
```javascript
const isActive = true; // Or some condition
if (isActive) {
  $('.my-element').addClass('active');
} else {
  $('.my-element').removeClass('active');
}
// Or for toggling based on existence:
$('.my-element').toggleClass('active');
```

❌ BAD: Relying on auto-execution of scripts in Ajax (removed in 4.0)
```javascript
// If fetching a script, this will no longer auto-execute in jQuery 4.0
$.ajax({
  url: '/my-script.js'
});
```

✅ GOOD: Explicitly define `dataType: "script"` for script execution
```javascript
$.ajax({
  url: '/my-script.js',
  dataType: 'script' // Explicitly tells jQuery to execute the script
});
```

## 5. Accessibility

While jQuery itself doesn't directly provide accessibility features, ensure your DOM manipulations respect ARIA attributes and semantic HTML.

❌ BAD: Relying solely on visual changes for state
```javascript
$('#my-button').on('click', function() {
  $(this).css('opacity', '0.5'); // Visually disabled, but not for screen readers
});
```

✅ GOOD: Updating ARIA attributes for accessibility
```javascript
$('#my-button').on('click', function() {
  $(this).attr('aria-disabled', 'true').prop('disabled', true).addClass('disabled-style');
});
```

## 6. WordPress Specifics

If working within a WordPress environment, always use the bundled jQuery version and respect its `noConflict` mode.

❌ BAD: Bundling your own jQuery version in a WordPress theme
```html
<script src="/path/to/my/jquery-3.7.1.min.js"></script>
```

✅ GOOD: Enqueue WordPress's bundled jQuery
```php
// In functions.php
wp_enqueue_script('my-script', get_template_directory_uri() . '/js/my-script.js', array('jquery'), '1.0.0', true);
```
Then, use the IIFE wrapper as described in 1.1 for your `my-script.js` file.
```javascript
// my-script.js
(function($) {
  // Your jQuery code here, $ is safely aliased to jQuery
  $('#wp-element').on('click', function() {
    console.log('WordPress element clicked!');
  });
})(jQuery);
```