---
description: This rule file provides comprehensive guidelines for writing high-quality Sphinx documentation, covering code style, structure, performance, and best practices. It aims to ensure consistency, readability, and maintainability of Sphinx-based projects.
globs: *.rst
---
- **Use reStructuredText (reST) or MyST markdown:** These are the primary markup languages for Sphinx. reST is the default and provides powerful semantic markup capabilities. MyST is a markdown flavor compatible with Sphinx. Choose one and stick to it for consistency.

- **Follow PEP 8 and PEP 257:** Adhere to PEP 8 for Python code style within your documentation, emphasizing readability and consistency (e.g., 4 spaces for indentation, line length limits). Follow PEP 257 for docstring conventions for Python code.

- **Structure Your Documentation Hierarchically:** Organize documentation into a clear, hierarchical structure with logical sections.  Use proper headings (H1, H2, etc.) to delineate content.

- **Use Semantic Markup:** Leverage reST's semantic markup for functions, classes, modules, and other code elements to enable cross-referencing and automatic linking. Example: `:func:`my_function``.

- **Cross-Referencing:** Utilize Sphinx's cross-referencing capabilities extensively to link related documentation sections, functions, classes, and other elements. This improves navigation and understanding.  Use `:ref:`, `:func:`, `:class:`, `:mod:` roles.

- **Document Code:** Use Sphinx's `autodoc` extension to automatically generate documentation from Python docstrings. Ensure your docstrings are comprehensive, following PEP 257 conventions. Include parameters, return types, and exceptions.

- **Use Code Blocks:** Clearly define code blocks using the `.. code-block:: language` directive to enable syntax highlighting and improve readability.

- **Whitespace and Formatting:** Maintain consistent whitespace for clarity.  Use blank lines to separate sections and ensure code blocks are clearly defined.

- **Maintain Line Length:** Limit line lengths to 79 characters for code and 72 characters for docstrings/comments to improve readability and compatibility with various tools and editors.

- **Version Control:** Use version control (e.g., Git) to manage your Sphinx documentation project. This allows for tracking changes, collaboration, and easy rollback to previous versions.

- **Use a Sphinx Theme:** Choose a Sphinx theme that suits your project's style and audience.  Many built-in themes (alabaster, classic, sphinxdoc) and third-party themes (sphinx_rtd_theme, pydata_sphinx_theme) are available.

- **Configure Sphinx Properly:** Carefully configure Sphinx using the `conf.py` file to customize the output format, extensions, theme, and other settings. Use meaningful project metadata (name, version, author).

- **Test Your Documentation:** Use Sphinx's `doctest` extension to automatically test code snippets within your documentation.  This ensures the code examples are correct and up-to-date.

- **Use Extensions:** Explore and utilize Sphinx extensions to enhance your documentation with features like mathematical equations, diagrams, and other specialized content.

- **Internationalization:** Consider internationalizing your documentation using Sphinx's internationalization features if your project targets a global audience.

- **Linkcheck Builder:** Use the linkcheck builder to automatically check for broken links in your documentation. This ensures the documentation remains up-to-date and accurate.

- **Contribute Extensions**: Create your own sphinx extensions to manage repeated formatting or add complex rendering.

- **Naming Conventions**
  - Use descriptive and consistent names for files, directories, and variables.
  - Follow Python's naming conventions for modules, classes, functions, and variables.
  - Use all lowercase for modules. Use underscores if it improves readability. Example: `my_module.py`
  - Use CapWords for class names. Example: `MyClass`
  - Use lowercase with underscores for functions and variables. Example: `my_function`, `my_variable`
  - Use UPPERCASE with underscores for constants. Example: `MAX_VALUE`

- **Common Patterns and Anti-Patterns**
  - **Pattern:** Use the Model-View-Controller (MVC) pattern for complex documentation structures, where the Model represents the data, the View renders the data, and the Controller handles user interactions.
  - **Anti-Pattern:** Avoid deeply nested directory structures, which can make it difficult to navigate and maintain the documentation.
  - **Anti-Pattern:** Avoid inconsistent formatting and style throughout the documentation, which can make it look unprofessional and difficult to read.

- **Performance Considerations**
  - Optimize images by compressing them without sacrificing quality.
  - Use lazy loading for large images or other resources that are not immediately visible on the page.
  - Use caching to reduce the number of times that resources need to be loaded.
  - Minify CSS and JavaScript files to reduce their size.

- **Security Best Practices**
  - Sanitize user inputs to prevent cross-site scripting (XSS) attacks.
  - Use HTTPS to encrypt communication between the user's browser and the server.
  - Keep Sphinx and its extensions up-to-date to patch security vulnerabilities.

- **Testing Approaches**
  - Write unit tests for custom Sphinx extensions.
  - Use integration tests to verify that the different parts of the documentation work together correctly.
  - Use end-to-end tests to simulate user interactions with the documentation and verify that the documentation behaves as expected.

- **Common Pitfalls and Gotchas**
  - Be aware of the different versions of Sphinx and their compatibility with different Python versions and extensions.
  - Use the appropriate encoding for your documentation files (UTF-8 is recommended).
  - Be careful when using the `raw` directive, as it can introduce security vulnerabilities if not used properly.

- **Tooling and Environment**
  - Use a code editor or IDE with syntax highlighting and linting for reStructuredText or Markdown.
  - Use a build automation tool (e.g., Make, tox) to automate the documentation build process.
  - Use a CI/CD system (e.g., Jenkins, Travis CI, GitHub Actions) to automatically build and deploy the documentation whenever the code changes.

- **Deployment Best Practices**
  - Use a dedicated server or hosting platform for your documentation.
  - Use a content delivery network (CDN) to distribute your documentation to users around the world.
  - Use HTTPS to secure your documentation.
  - Use a robots.txt file to prevent search engines from indexing sensitive parts of your documentation.

- **Documenting Objects**
  - Use `autodoc` to automatically generate documentation for Python objects.
  - Use directives like `.. automodule::`, `.. autoclass::`, and `.. autofunction::` to document modules, classes, and functions, respectively.
  - Use the `:members:`, `:undoc-members:`, and `:show-inheritance:` options to control which members are documented.
  - Write clear and concise docstrings for all objects, following PEP 257 conventions.

- **Intersphinx**
  - Use the `intersphinx` extension to link to documentation of other Sphinx projects.
  - Configure `intersphinx_mapping` in `conf.py` to specify the locations of other Sphinx documentation.
  - Use the `:py:mod:`, `:py:class:`, and `:py:func:` roles to link to modules, classes, and functions in other projects.

- **reStructuredText Specific Tips**
  - Use titles and sections to structure your documentation.
  - Use bullet lists and numbered lists to present information in a clear and concise way.
  - Use tables to organize data.
  - Use images to illustrate concepts.
  - Use footnotes and citations to provide additional information and give credit to sources.

- **Example `conf.py` settings:**
  python
  # Configuration file for the Sphinx documentation builder.
  # For the full list of built-in configuration values, see the documentation:
  # https://www.sphinx-doc.org/en/master/usage/configuration.html

  # -- Project information -----------------------------------------------------

  project = 'My Project'
  copyright = '2023, My Company'
  author = 'John Doe'
  release = '1.0.0'

  # -- General configuration ---------------------------------------------------

  extensions = [
      'sphinx.ext.autodoc',
      'sphinx.ext.napoleon',
      'sphinx.ext.intersphinx',
      'sphinx.ext.todo',
      'sphinx.ext.viewcode',
      'sphinx.ext.mathjax',
      'sphinx.ext.ifconfig',
      'sphinx.ext.githubpages',
      'sphinx.ext.graphviz'
  ]

  templates_path = ['_templates']
  exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

  language = 'en'

  # -- Options for HTML output -------------------------------------------------

  html_theme = 'sphinx_rtd_theme'
  html_static_path = ['_static']

  # -- Extension configuration -------------------------------------------------

  # napoleon configuration (for Google-style docstrings)
  napoleon_google_docstring = True
  napoleon_include_init_with_doc = True
  napoleon_include_private_with_doc = False
  napoleon_include_special_with_doc = True
  napoleon_use_rtype = True
  napoleon_use_param = True

  # intersphinx configuration
  intersphinx_mapping = {
      'python': ('https://docs.python.org/3', None),
      'numpy': ('https://numpy.org/doc/stable/', None),
  }

  # todo configuration
  todo_include_todos = True

  # graphviz configuration
  graphviz_output_format = 'svg'


  # -- Options for Markdown files --------------------------------------------

  from recommonmark.parser import CommonMarkParser

source_parsers = {
    '.md': CommonMarkParser,
}

  source_suffix = ['.rst', '.md']