---
description: Definitive guide for building modern, maintainable Kivy applications using best practices, focusing on KV language, KivyMD, and clear code organization.
globs: **/*.py
---
# Kivy Best Practices

This guide outlines the definitive best practices for developing Kivy applications. Adhere to these rules to ensure your codebase is maintainable, performant, and scales effectively.

## 1. Prioritize KV Language for UI Definition

**Opinion:** Always separate your UI definition from Python logic using Kivy Language (`.kv` files). This is the cornerstone of maintainable Kivy applications and enables declarative UI. Avoid building complex widget trees imperatively in Python.

**When to use:** For all visual components, layouts, bindings, and animations.

**❌ BAD (Imperative Python UI):**
```python
# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class BadApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        button1 = Button(text='Click Me 1')
        button2 = Button(text='Click Me 2')
        layout.add_widget(button1)
        layout.add_widget(button2)
        return layout

BadApp().run()
```

**✅ GOOD (Declarative KV Language):**
```python
# main.py
from kivy.app import App
from kivy.lang import Builder

class GoodApp(App):
    def build(self):
        # Kivy automatically loads 'goodapp.kv' if not specified
        return Builder.load_file('goodapp.kv')

GoodApp().run()
```
```kv
# goodapp.kv
BoxLayout:
    orientation: 'vertical'
    Button:
        text: 'Click Me 1'
    Button:
        text: 'Click Me 2'
```

## 2. Leverage KivyMD for Material Design

**Opinion:** For modern, polished UIs, use KivyMD. It provides a rich set of Material Design components that integrate seamlessly with Kivy's KV language.

**When to use:** For all new UI development requiring a modern aesthetic.

**❌ BAD (Vanilla Kivy for complex components):**
```kv
# my_screen.kv
BoxLayout:
    # ... manually style a button to look like Material Design ...
    Button:
        background_color: 0.1, 0.5, 0.8, 1
        color: 1, 1, 1, 1
        padding: dp(16), dp(8)
        # ... many more properties ...
```

**✅ GOOD (KivyMD components):**
```kv
# my_screen.kv
MDBoxLayout: # Use MD layouts for consistency
    orientation: 'vertical'
    MDTopAppBar:
        title: "KivyMD App"
    MDButton:
        MDButtonText:
            text: "Material Button"
        on_release: app.root.ids.label.text = "Button pressed!"
    MDLabel:
        id: label
        text: "Hello KivyMD"
        halign: 'center'
```
```python
# main.py
from kivymd.app import MDApp
from kivy.lang import Builder

class MDExampleApp(MDApp):
    def build(self):
        # Load KV string directly for simple examples, or use Builder.load_file('my_screen.kv')
        return Builder.load_string("""
MDBoxLayout:
    orientation: 'vertical'
    MDTopAppBar:
        title: "KivyMD App"
    MDLabel:
        text: "Welcome to KivyMD!"
        halign: 'center'
""")

MDExampleApp().run()
```

## 3. Structure Your Project Modularly

**Opinion:** Organize your Kivy project into logical directories for maintainability and scalability. Separate KV files, Python logic, screens, and custom widgets.

**When to use:** For any project beyond a simple single-file example.

**❌ BAD (Monolithic structure):**
```
my_app/
├── main.py
├── all_screens_and_widgets.kv
├── images/
```

**✅ GOOD (Modular structure):**
```
my_app/
├── main.py
├── screens/
│   ├── home_screen.py
│   └── home_screen.kv
│   ├── settings_screen.py
│   └── settings_screen.kv
├── widgets/
│   ├── custom_button.py
│   └── custom_button.kv
├── kv/ # For global styles or shared rules
│   └── styles.kv
├── assets/
│   ├── images/
│   └── fonts/
```

## 4. Use Kivy Properties for Reactive UI

**Opinion:** Always use Kivy's `Property` types (`StringProperty`, `NumericProperty`, `ObjectProperty`, `BooleanProperty`, `ListProperty`, `DictProperty`) for attributes that need to trigger UI updates or be bound in KV language. Avoid plain Python attributes for reactive data.

**When to use:** For any data that affects the UI or needs to be observed.

**❌ BAD (Plain Python attributes):**
```python
# my_widget.py
from kivy.uix.boxlayout import BoxLayout

class MyWidget(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.my_text = "Initial Text" # Will not update UI automatically
```
```kv
# my_widget.kv
<MyWidget>:
    Label:
        text: root.my_text # Won't update if root.my_text changes in Python
```

**✅ GOOD (Kivy Properties):**
```python
# my_widget.py
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty

class MyWidget(BoxLayout):
    my_text = StringProperty("Initial Text") # Will update UI automatically

    def update_text(self, new_text):
        self.my_text = new_text
```
```kv
# my_widget.kv
<MyWidget>:
    orientation: 'vertical'
    Label:
        text: root.my_text # Updates automatically
    Button:
        text: "Change Text"
        on_release: root.update_text("New Text from Button")
```

## 5. Adhere to Kivy Coding Style

**Opinion:** Follow PEP 8 with Kivy-specific adjustments for consistency and readability.

**When to use:** Always.

- **Docstrings/Comments:** Limit to 79 characters.
- **Quotes:** Use single quotes for identifiers (e.g., `'horizontal'`, `'center_x'`). Use double quotes for user-facing strings (e.g., `"Hello World"`). Escape characters as needed.

**❌ BAD (Inconsistent quoting, long comments):**
```python
# Bad example
def my_function(orientation="vertical"): # This is a very long comment that exceeds the recommended line length for docstrings and comments in Kivy projects, making the code harder to read and maintain.
    print(f"Orientation is {orientation}")
```

**✅ GOOD (Kivy style):**
```python
# Good example
def my_function(orientation='vertical'):
    """
    Prints the given orientation.
    """
    print(f"Orientation is {orientation}")

# In KV
Button:
    text: "Submit" # User-facing string
    orientation: 'horizontal' # Identifier
```

## 6. Manage Widget IDs Effectively

**Opinion:** Use `ids` in KV for referencing child widgets within the same KV scope. For cross-scope communication or complex state management, bind to Kivy Properties or use event dispatching. Avoid over-reliance on `ids` in Python code for direct widget manipulation, especially across different screens or deeply nested structures.

**When to use:**
- `ids`: For simple interactions within a single KV file or custom widget's KV.
- Kivy Properties/Events: For robust communication between Python logic and UI, or between different components.

**❌ BAD (Over-reliance on `ids` in Python, hard-to-trace dependencies):**
```python
# main.py
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

class HomeScreen(Screen):
    pass

class MyManager(ScreenManager):
    pass

class BadApp(App):
    def build(self):
        return Builder.load_string("""
MyManager:
    HomeScreen:
        name: 'home'
        BoxLayout:
            Button:
                id: my_button
                text: 'Press Me'
""")
    def on_start(self):
        # Fragile: relies on deep knowledge of widget tree and IDs
        self.root.get_screen('home').ids.my_button.text = "Updated!"

BadApp().run()
```

**✅ GOOD (Using Kivy Properties for state, `ids` for local KV binding):**
```python
# main.py
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import StringProperty

class HomeScreen(Screen):
    button_text = StringProperty("Press Me")

class MyManager(ScreenManager):
    pass

class GoodApp(App):
    def build(self):
        return Builder.load_string("""
MyManager:
    HomeScreen:
        name: 'home'
        BoxLayout:
            Button:
                text: root.button_text # Bind to property
                on_release: app.update_home_button()
""")
    def update_home_button(self):
        home_screen = self.root.get_screen('home')
        home_screen.button_text = "Updated via Property!"

GoodApp().run()
```

## 7. Optimize Layouts for Responsiveness

**Opinion:** Use `BoxLayout`, `GridLayout`, and `FloatLayout` (or their KivyMD equivalents like `MDBoxLayout`) for responsive UI design. Understand their strengths and weaknesses.

**When to use:**
- `BoxLayout` / `MDBoxLayout`: For linear arrangements (horizontal/vertical). Excellent for toolbars, navigation, or stacking elements.
- `GridLayout` / `MDGridLayout`: For grid-based arrangements where items have similar sizes.
- `FloatLayout` / `MDFloatLayout`: For absolute positioning or layering widgets, often combined with `pos_hint` and `size_hint`. Use sparingly for main layouts, prefer for overlays or specific components.

**❌ BAD (Hardcoded sizes/positions, non-responsive):**
```kv
# my_screen.kv
FloatLayout:
    Button:
        size: 100, 50 # Fixed size
        pos: 50, 50 # Fixed position
    Label:
        text: "Hello"
        size: 200, 30
        pos: 150, 100
```

**✅ GOOD (Responsive layouts with hints):**
```kv
# my_screen.kv
MDBoxLayout:
    orientation: 'vertical'
    MDTopAppBar:
        title: "Responsive App"
    MDBoxLayout:
        orientation: 'horizontal'
        spacing: dp(10)
        padding: dp(10)
        MDButton:
            MDButtonText:
                text: "Button 1"
            size_hint_x: 0.3 # Takes 30% of horizontal space
        MDLabel:
            text: "Content Area"
            halign: 'center'
            size_hint_x: 0.7 # Takes 70% of horizontal space
```

## 8. Implement Type Hints

**Opinion:** Use Python type hints (`typing` module) for all Kivy Python code. This improves code readability, enables static analysis, and reduces runtime errors.

**When to use:** For function arguments, return values, and class attributes (especially Kivy Properties).

**❌ BAD (Untyped Python):**
```python
# my_widget.py
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty

class MyWidget(BoxLayout):
    my_text = StringProperty("Initial Text")

    def update_text(self, new_text):
        self.my_text = new_text
```

**✅ GOOD (Typed Python):**
```python
# my_widget.py
from typing import Union
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ObjectProperty
from kivy.event import EventDispatcher

class MyWidget(BoxLayout):
    my_text: StringProperty = StringProperty("Initial Text")
    # Type hint for a Kivy Property that holds an instance of another widget
    linked_widget: ObjectProperty = ObjectProperty(None)

    def update_text(self, new_text: str) -> None:
        self.my_text = new_text

    def set_linked_widget(self, widget: EventDispatcher) -> None:
        self.linked_widget = widget
```

## 9. Performance Considerations

**Opinion:** Profile your KV language parsing with `KIVY_PROFILE_LANG=1` during development. Avoid excessive nesting or complex calculations in KV where Python might be more efficient. Use `RecycleView` for large lists.

**When to use:**
- `KIVY_PROFILE_LANG`: During development and optimization phases.
- `RecycleView`: For displaying large, dynamic lists of items to maintain smooth scrolling performance.

**❌ BAD (Slow list rendering):**
```kv
# my_screen.kv
ScrollView:
    BoxLayout: # Creates all widgets at once, even if not visible
        orientation: 'vertical'
        size_hint_y: None
        height: self.minimum_height
        # ... many, many labels or buttons ...
```

**✅ GOOD (Efficient list rendering with RecycleView):**
```kv
# my_screen.kv
#:import Factory kivy.factory.Factory # Needed if ListItem is not defined in Python
<ListItem@BoxLayout>: # Define a view class for RecycleView
    size_hint_y: None
    height: dp(48)
    Label:
        id: rv_label
        text: root.text_item

RecycleView:
    id: rv
    viewclass: 'ListItem'
    RecycleBoxLayout:
        default_size: None, dp(48)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
```
```python
# main.py
from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout # Needed for ListItem base class

class ListItem(BoxLayout): # Define the view class in Python
    text_item = StringProperty('')

class RecycleViewApp(MDApp):
    def build(self):
        return Builder.load_string("""
#:import ListItem __main__.ListItem # Import the Python class

MDBoxLayout:
    orientation: 'vertical'
    MDTopAppBar:
        title: "RecycleView Example"
    RecycleView:
        id: rv
        viewclass: 'ListItem'
        RecycleBoxLayout:
            default_size: None, dp(48)
            default_size_hint: 1, None
            size_hint_y: None
            height: self.minimum_height
            orientation: 'vertical'
""")

    def on_start(self):
        # Populate RecycleView with data
        data = [{'text_item': f'Item {i}'} for i in range(1000)]
        self.root.ids.rv.data = data

RecycleViewApp().run()
```

## 10. Accessibility Considerations

**Opinion:** While Kivy's direct accessibility features are evolving, ensure your UI is navigable and understandable for all users. Provide clear labels, sufficient contrast, and logical focus order.

**When to use:** In all UI design and implementation.

- **Labels:** Always provide clear, descriptive text for interactive elements.
- **Contrast:** Ensure sufficient color contrast between text and backgrounds.
- **Focus Order:** Design layouts that allow for a logical tab/focus order, even if Kivy doesn't fully automate it. Consider adding custom keyboard navigation if needed.

**❌ BAD (Ambiguous UI):**
```kv
# my_screen.kv
Button:
    text: '' # Empty text, relies on icon only
    Image:
        source: 'info_icon.png'
```

**✅ GOOD (Clear labels, descriptive text):**
```kv
# my_screen.kv
MDButton:
    MDButtonText:
        text: "Show Info" # Clear text label
    MDButtonIcon:
        icon: "information-outline"
    on_release: app.show_info_dialog()
```