---
description: Definitive guide for building modern, performant, and maintainable Pygame applications using `pygame-gui` for UI, focusing on structure, responsive layout, theme-driven styling, and performance best practices.
globs: **/*.py
---
# Pygame Best Practices

This guide outlines the essential patterns for building robust and visually appealing Pygame applications, leveraging `pygame-gui` for all UI interactions. Adhere to these principles for maintainable, performant, and scalable game development.

## 1. Core Application Structure

Always structure your Pygame application with a clear main loop, a single `UIManager`, and a fixed timestep. Separate game logic from UI logic.

❌ BAD: Scattered UI element creation, no `UIManager`, variable framerate.

```python
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Bad Example")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # UI logic directly in main loop
        if event.type == pygame.MOUSEBUTTONDOWN:
            print("Mouse clicked!")

    screen.fill((0, 0, 0))
    # Draw game elements
    pygame.draw.rect(screen, (255, 0, 0), (100, 100, 50, 50))
    pygame.display.flip()
```

✅ GOOD: Dedicated `UIManager`, fixed timestep, clear event processing.

```python
import pygame
import pygame_gui
from pygame_gui.core.interfaces import IUIManagerInterface

class Game:
    def __init__(self, screen_size: tuple[int, int]):
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Good Example")
        self.manager: IUIManagerInterface = pygame_gui.UIManager(screen_size, 'data/themes/default.json')
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.fps = 60

        # Example UI button
        pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 275), (100, 50)),
            text='Click Me',
            manager=self.manager
        )

    def run(self):
        while self.is_running:
            time_delta = self.clock.tick(self.fps) / 1000.0
            self._handle_events()
            self._update(time_delta)
            self._draw()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                print("Button pressed!")
            self.manager.process_events(event)

    def _update(self, time_delta: float):
        self.manager.update(time_delta)
        # Update game logic here

    def _draw(self):
        self.screen.fill((0, 0, 0))
        # Draw game elements
        pygame.draw.rect(self.screen, (255, 0, 0), (100, 100, 50, 50))
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

if __name__ == '__main__':
    game = Game((800, 600))
    game.run()
```

## 2. UI Management with `pygame-gui`

`pygame-gui` is the definitive choice for polished, interactive UIs.

### 2.1. Initialize `UIManager` Once

Create a single `UIManager` instance early in your application lifecycle.

```python
# In your Game class or main setup
self.manager: IUIManagerInterface = pygame_gui.UIManager(screen_size, 'data/themes/default.json')
```

### 2.2. Responsive Layouts with `relative_rect` and Anchors

Always use `relative_rect` for UI element positioning. Leverage `anchors` for responsive designs that adapt to window resizing and container movement. Avoid hardcoding absolute pixel positions.

❌ BAD: Hardcoded absolute positions, no responsiveness.

```python
# Button position will be fixed regardless of window size or parent container
pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(50, 50, 100, 30), # Absolute position
    text='Button', manager=self.manager
)
```

✅ GOOD: Relative positioning and anchors for dynamic layouts.

```python
# Button anchored to the bottom-right of its container, with a 10px offset
pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(0, 0, 100, 30),
    text='Bottom Right',
    manager=self.manager,
    anchors={'right': 'right', 'bottom': 'bottom',
             'left': 'right', 'top': 'bottom'} # Stretch to maintain size
).set_relative_position((-110, -40)) # Offset from bottom-right corner

# Button centered in its container
pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect(0, 0, 150, 50),
    text='Centered',
    manager=self.manager,
    anchors={'center': 'center'}
)
```

### 2.3. Theme-Driven Styling with JSON

Externalize all UI styling (colors, fonts, sizes) into JSON theme files. This enables rapid visual iteration without code changes.

❌ BAD: Hardcoded colors and fonts throughout the code.

```python
# In your code
pygame_gui.elements.UIButton(
    ...
    text='Red Button',
    manager=self.manager,
    object_id='#red_button' # Still needs object_id for theming, but no theme loaded
)
# No theme loaded, so default styles apply or you'd set them programmatically.
```

✅ GOOD: Load a JSON theme file and use `object_id` for specific styling.

```json
// data/themes/default.json
{
    "button": {
        "colours": {
            "normal_bg": "#25292e",
            "hovered_bg": "#35393e",
            "pressed_bg": "#15191e",
            "normal_text": "#FFFFFF"
        },
        "font": {
            "name": "fira_code",
            "size": "20",
            "bold": "0",
            "italic": "0"
        }
    },
    "#red_button": {
        "colours": {
            "normal_bg": "#FF0000"
        }
    }
}
```

```python
# In your Game class or main setup
self.manager: IUIManagerInterface = pygame_gui.UIManager(screen_size, 'data/themes/default.json')

# In your UI creation
pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 10), (100, 30)),
    text='Themed Button',
    manager=self.manager,
    object_id='#red_button' # Applies styles from the theme file
)
```

## 3. Performance Considerations

### 3.1. Fixed Timestep

Always use `pygame.time.Clock().tick(FPS)` to cap your frame rate and calculate `time_delta`. This ensures consistent game speed across different hardware and is crucial for `pygame-gui` updates.

❌ BAD: No frame rate cap, variable `time_delta`.

```python
# ...
while is_running:
    # No clock.tick(), game runs as fast as possible
    # time_delta would need to be calculated differently or omitted, leading to issues
    for event in pygame.event.get():
        # ...
    # ...
    pygame.display.update()
```

✅ GOOD: Consistent frame rate and `time_delta`.

```python
# ...
self.clock = pygame.time.Clock()
self.fps = 60
# ...
while self.is_running:
    time_delta = self.clock.tick(self.fps) / 1000.0 # Essential for consistent updates
    self.manager.update(time_delta) # UIManager relies on time_delta
    # ...
    pygame.display.update()
```

### 3.2. Avoid "Threaded Flip" Anti-pattern

Never call `pygame.display.flip()` or `pygame.display.update()` from a separate thread. All drawing operations must occur on the main thread to prevent crashes and undefined behavior.

## 4. Component Architecture

Encapsulate complex UI elements or game screens into their own classes. Each screen/component should manage its own `pygame-gui` elements and event handling.

```python
from pygame_gui.core.interfaces import IUIManagerInterface

class MainMenuScreen:
    def __init__(self, manager: IUIManagerInterface, screen_rect: pygame.Rect):
        self.manager = manager
        self.container = pygame_gui.elements.UIPanel(
            relative_rect=screen_rect,
            manager=manager,
            anchors={'left': 'left', 'right': 'right', 'top': 'top', 'bottom': 'bottom'}
        )
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, 0, 200, 50),
            text='Start Game',
            manager=self.manager,
            container=self.container,
            anchors={'center': 'center'}
        )
        self.exit_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(0, 60, 200, 50), # Offset from start_button
            text='Exit',
            manager=self.manager,
            container=self.container,
            anchors={'centerx': 'centerx', 'top_target': self.start_button} # Anchor to another element
        )

    def process_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.start_button:
                print("Start Game Pressed!")
                return True
            if event.ui_element == self.exit_button:
                print("Exit Game Pressed!")
                return True
        return False

    def hide(self):
        self.container.hide()

    def show(self):
        self.container.show()
```

## 5. Type Hints

Always use type hints for clarity, maintainability, and to leverage static analysis tools. This is especially important for `pygame` and `pygame-gui` objects.

```python
import pygame
from pygame_gui.core.interfaces import IUIManagerInterface
from typing import Tuple

def create_button(manager: IUIManagerInterface, rect: pygame.Rect, text: str) -> pygame_gui.elements.UIButton:
    """Creates a standard UI button."""
    return pygame_gui.elements.UIButton(relative_rect=rect, text=text, manager=manager)

def get_screen_size(screen: pygame.Surface) -> Tuple[int, int]:
    """Returns the width and height of the screen."""
    return screen.get_width(), screen.get_height()
```

## 6. `pygame-gui` vs `pygame-menu`

*   **`pygame-gui`**: Use for complex, interactive, in-game UIs (inventories, settings panels, dialogues). Its robust layout, theming, and event system are ideal for rich interfaces.
*   **`pygame-menu`**: Use for simple, quick-to-implement menus like start screens, pause overlays, or basic option screens where `pygame-gui` might be overkill. It offers a simpler API for common menu patterns.

For most projects, `pygame-gui` is the recommended default due to its flexibility and modern feature set.

## 7. Accessibility

Design your UI with accessibility in mind. Use clear, high-contrast themes, ensure interactive elements are sufficiently large, and provide keyboard navigation where appropriate. `pygame-gui` supports keyboard navigation and theme-driven text scaling, which are foundational for accessibility.

## 8. PEP 8 Adherence

Follow PEP 8 style guidelines rigorously. Consistent code is readable code. Use linters (e.g., Flake8, Black) to enforce this automatically.