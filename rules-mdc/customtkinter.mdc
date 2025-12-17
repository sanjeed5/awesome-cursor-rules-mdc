---
description: Definitive guidelines for writing clean, maintainable, and performant customtkinter applications. This rule file enforces best practices for code structure, theming, component architecture, and common pitfalls.
globs: **/*.py
---
# customtkinter Best Practices

This document outlines the definitive best practices for developing applications with `customtkinter`. Adhering to these guidelines ensures your UI code is consistent, maintainable, performant, and leverages the full power of the library.

## 1. Code Organization: Class-Based Applications

Always structure your `customtkinter` applications using a class that inherits from `customtkinter.CTk`. This encapsulates UI logic, promotes modularity, and simplifies state management.

❌ BAD: Procedural spaghetti code
```python
import customtkinter as ctk

# Theme setup (often forgotten or inconsistent)
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("400x240")
app.title("My App")

def on_button_click():
    print("Button clicked!")
    # Logic mixed directly here

label = ctk.CTkLabel(app, text="Hello!")
label.pack(pady=10)

button = ctk.CTkButton(app, text="Click Me", command=on_button_click)
button.pack(pady=10)

app.mainloop()
```

✅