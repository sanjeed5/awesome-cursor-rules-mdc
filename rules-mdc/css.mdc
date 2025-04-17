---
description: This rule provides best practices for CSS development, covering code organization, performance, security, testing, and common pitfalls. It aims to ensure maintainable, scalable, and efficient CSS code.
globs: *.css
---
- **Use a CSS Preprocessor**: Leverage preprocessors like Sass, Less, or Stylus for features like variables, nesting, mixins, and functions to enhance code organization and maintainability.

- **Code Organization and Structure**:
  - **Directory Structure**: Organize CSS files into logical directories based on functionality (e.g., `components`, `modules`, `pages`, `themes`).
  - **File Naming Conventions**: Use meaningful and consistent file names (e.g., `button.css`, `header.module.css`). Consider using a naming convention like BEM (Block, Element, Modifier) or SUIT CSS.
  - **Module Organization**: Break down large stylesheets into smaller, reusable modules.
  - **Component Architecture**: Structure CSS based on UI components, ensuring each component has its dedicated stylesheet.
  - **Code Splitting**: Split CSS into critical (above-the-fold) and non-critical CSS for improved initial page load time. Utilize tools like `Critical CSS` or `PurgeCSS`.

- **Formatting and Style**:
  - **Indentation**: Use soft-tabs with two spaces for indentation.
  - **Trailing Whitespace**: Avoid trailing whitespace.
  - **Declaration Order**: Maintain a consistent declaration order (e.g., alphabetical, grouping related properties).
  - **Property Units**: Use appropriate units (e.g., `rem` or `em` for font sizes, `vw` or `%` for responsive layouts).
  - **Quotation**: Use double quotes for attribute values in selectors.
  - **Line Length**: Keep lines reasonably short (e.g., under 80-100 characters) for readability.

- **Common Patterns and Anti-patterns**:
  - **BEM (Block, Element, Modifier)**: Use BEM or similar methodologies for clear and maintainable class naming.
  - **Object-Oriented CSS (OOCSS)**: Apply OOCSS principles to create reusable and scalable styles.
  - **Avoid !important**: Minimize the use of `!important` to prevent specificity conflicts.
  - **Avoid Inline Styles**: Prefer external stylesheets or embedded styles over inline styles for better maintainability and separation of concerns.
  - **Don't use IDs for styling**: Avoid using IDs for styling since they are too specific and make styles harder to override.

- **Understanding CSS Specificity**: Master CSS specificity rules to avoid unexpected style conflicts. Use specific selectors sparingly.

- **Use Flexible/Relative Units**: Employ relative units like `em`, `rem`, and `vw` for creating responsive designs that adapt to different screen sizes.

- **Performance Considerations**:
  - **Optimize Selectors**: Use efficient CSS selectors to minimize rendering time. Avoid overly complex selectors.
  - **Minify CSS**: Minify CSS files to reduce file size and improve loading times.
  - **Compress CSS**: Use Gzip or Brotli compression on the server to further reduce CSS file sizes.
  - **Browser Caching**: Leverage browser caching to store CSS files locally, reducing server load and improving performance.
  - **Avoid Expensive Properties**: Avoid CSS properties that are computationally expensive (e.g., `filter`, `box-shadow` with large blur radii) when possible.

- **Security Best Practices**:
  - **Sanitize User Input**: When using CSS variables based on user input, sanitize the input to prevent CSS injection attacks.
  - **Content Security Policy (CSP)**: Implement a CSP to control the sources from which CSS can be loaded.

- **Testing Approaches**:
  - **Unit Testing**: Use tools like `CSS Modules` or `styled-components` to write unit tests for individual CSS components.
  - **Visual Regression Testing**: Implement visual regression testing to detect unexpected changes in CSS styles.
  - **Linting**: Use a CSS linter (e.g., `stylelint`) to enforce coding standards and catch potential errors.

- **Common Pitfalls and Gotchas**:
  - **Specificity Conflicts**: Be aware of specificity issues and use tools to visualize and manage CSS specificity.
  - **Browser Compatibility**: Test CSS across different browsers and versions to ensure compatibility.
  - **Vendor Prefixes**: Use vendor prefixes when necessary but consider using tools like `autoprefixer` to automate the process.

- **Tooling and Environment**:
  - **CSS Linters**: Use `stylelint` to enforce coding standards and identify potential errors.
  - **CSS Formatters**: Employ tools like `Prettier` to automatically format CSS code consistently.
  - **Build Tools**: Integrate CSS compilation, minification, and optimization into your build process using tools like `Webpack`, `Parcel`, or `Gulp`.
  - **CSS Modules**: Use CSS Modules to scope CSS classes locally and avoid naming conflicts.

- **Additional Tips**:
  - **Document CSS**: Add comments to explain complex or non-obvious CSS rules.
  - **Keep CSS DRY (Don't Repeat Yourself)**: Reuse CSS code as much as possible using variables, mixins, and inheritance.
  - **Regularly Review and Refactor**: Regularly review and refactor CSS code to maintain its quality and prevent code bloat.
  - **Consider Accessibility**: Ensure CSS styles contribute to website accessibility (e.g., sufficient color contrast, proper use of semantic HTML).
  - **Use CSS Variables (Custom Properties)**: Use CSS variables for theming and to manage values across your CSS.