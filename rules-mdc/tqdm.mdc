---
description: This rule file provides best practices and coding standards for using the `tqdm` library in Python. It focuses on performance, customization, and avoiding common pitfalls.
globs: *.py
---
# tqdm Best Practices and Coding Standards

This document outlines best practices for using the `tqdm` library in Python, focusing on simplicity, customization, performance optimization, and avoiding common pitfalls. Adhering to these guidelines will help you create efficient and informative progress bars for your projects.

## Library Information

- Name: tqdm
- Tags: python, utilities, progress-bar, command-line

## 1. Basic Usage and Integration

- **Wrap Iterables Directly:** The simplest and most common way to use `tqdm` is by wrapping your iterable object directly with the `tqdm()` function.

  python
  from tqdm import tqdm

  for item in tqdm(my_iterable):
      # Process item
      ...
  

- **Descriptive Progress Bars:** Always use the `desc` parameter to add a short, descriptive text to your progress bar, providing context to the user.

  python
  for item in tqdm(my_iterable, desc="Processing Data"):
      ...
  

- **Integration with Pandas:**  Use `tqdm` with Pandas `apply` functions for data analysis tasks.

  python
  import pandas as pd
  from tqdm import tqdm

  tqdm.pandas()
  df['column'].progress_apply(lambda x: some_function(x))
  

## 2. Performance Considerations

- **Update Frequency:** Avoid updating the progress bar too frequently, as it can significantly impact performance, especially with large datasets or computationally intensive tasks. Adjust `mininterval` and `maxinterval` to optimize refresh rates.

  python
  for item in tqdm(my_iterable, desc="Processing Data", mininterval=1, maxinterval=10):
      ...
  

- **Manual Control for Performance:** In scenarios where automatic iteration tracking isn't feasible, use manual control with `tqdm` to update the progress bar at strategic intervals.

  python
  from tqdm import tqdm
  import time

  total_iterations = 1000
  with tqdm(total=total_iterations, desc="Manual Progress") as pbar:
      for i in range(total_iterations):
          # Perform some operation
          time.sleep(0.001) # simulate work
          if i % 10 == 0:
              pbar.update(10) # update after every 10 iterations
  

- **`tqdm.write()` for Printing:**  Use `tqdm.write()` to print messages to the console without disrupting the progress bar. This is especially useful for logging information or displaying intermediate results.

  python
  from tqdm import tqdm

  for i in tqdm(range(100), desc='Processing'):
      if i % 10 == 0:
          tqdm.write(f'Iteration {i}: Some intermediate result')
  

## 3. Nested Progress Bars

- **`leave=False` for Nested Bars:** When using nested loops with `tqdm`, set `leave=False` for inner loops to prevent cluttering the output. This ensures that only the outer loop's progress bar remains after the inner loop completes.

  python
  from tqdm import tqdm
  import time

  for i in tqdm(range(5), desc="Outer Loop", leave=True):
      for j in tqdm(range(3), desc="Inner Loop", leave=False):
          time.sleep(0.1)
  

## 4. Customization and Advanced Features

- **Dynamic Descriptions:** Update the progress bar description dynamically during iterations to provide more context-specific information.

  python
  from tqdm import tqdm
  import time

  with tqdm(range(10), desc="Starting") as pbar:
      for i in pbar:
          time.sleep(0.5)
          pbar.set_description(f"Step {i+1} completed")
  

- **Custom Formatting:** Customize the appearance of the progress bar using `bar_format` to control the layout, colors, and displayed information.

  python
  from tqdm import tqdm
  import time

  for i in tqdm(range(5), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
      time.sleep(0.5)
  

- **GUI Mode (Jupyter Notebooks):** Use `tqdm.notebook` for a more visually appealing progress bar in Jupyter notebooks.  Import `tqdm.notebook` rather than `tqdm`.

  python
  from tqdm.notebook import tqdm
  import time

  for i in tqdm(range(1000)):
      time.sleep(0.001)
  

## 5. Common Pitfalls and Anti-Patterns

- **Over-Updating:** Updating the progress bar too frequently is a common mistake.  This can significantly slow down your code.  Adjust `mininterval` and `maxinterval`, or use manual updates.

- **Ignoring `leave=False` in Nested Loops:** Forgetting to set `leave=False` in nested loops can lead to cluttered output, making it difficult to read the progress of the outer loop.

- **Not Closing the Progress Bar:**  If you're using manual control, ensure you close the progress bar with `pbar.close()` to release resources.

- **Incorrect Total Value:** Providing an incorrect `total` value in the `tqdm()` constructor can lead to inaccurate progress display. Double-check the iterable's length.

- **Using `print()` Within the Loop:** Using the standard `print()` function within the loop can disrupt the progress bar display.  Use `tqdm.write()` instead.

## 6. Testing Approaches

- **Unit Tests:** When using `tqdm` in functions, test the function's core logic independently of `tqdm`. If you need to verify `tqdm` output, consider capturing standard output for assertions, though this is generally less valuable than testing the core function logic.

- **Integration Tests:** Ensure that `tqdm` integrates correctly with your data processing pipelines. Verify that the progress bars are displayed accurately and don't introduce unexpected performance bottlenecks.

## 7. Tooling and Environment

- **Development Tools:**  Use standard Python development tools like VS Code, PyCharm, or Jupyter Notebooks for working with `tqdm`.

- **Linting and Formatting:** Adhere to PEP 8 style guidelines and use linters like `flake8` or `pylint` to maintain code quality. Format your code with `black` or `autopep8` for consistency.

## 8. Example: Downloading Files with Progress

python
import requests
from tqdm import tqdm


def download_file(url, filename):
    """Downloads a file from a URL with a progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB
        with tqdm(desc=filename, total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        print(f"Download complete: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")


# Example usage:
file_url = "https://www.example.com/large_file.zip"  # Replace with an actual URL
file_name = "large_file.zip"
download_file(file_url, file_name)


## 9. Conclusion

By following these best practices, you can effectively leverage the `tqdm` library to create informative and efficient progress bars in your Python projects, improving the user experience and providing valuable insights into the execution of your code.