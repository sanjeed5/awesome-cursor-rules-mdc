---
description: Definitive guide for building modern, maintainable, and performant PySide6 applications using best practices like UI/logic separation, modern controls, and robust type safety.
globs: **/*.py
---
# PySide6 Best Practices

This guide outlines the essential best practices for developing robust, maintainable, and modern PySide6 applications. Adhere to these principles to ensure high-quality, performant, and future-proof code.

## 1. Code Organization & UI Generation

**Principle:** Strictly separate UI definition from application logic. Leverage Qt Designer for visual UI creation and `pyside6-uic` for generating Python UI classes.

**Rule:** Always design your user interfaces visually in Qt Designer. Convert the `.ui` files to Python classes using `pyside6-uic`, then import and compose these generated UI classes within a dedicated Python controller class. **Never manually modify the generated `ui_*.py` files.**

❌ **BAD:** Hand-coding complex UI layouts directly in Python, or modifying generated UI files.
```python
# main.py (Bad: Hand-coding UI directly)
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bad UI Design - Hand-coded")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.button = QPushButton("Click Me")
        layout.addWidget(self.button)
        self.button.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        print("Button clicked!")
```

✅ **GOOD:** Use `pyside6-uic` generated UI classes composed in a controller.
```python
# 1. Design 'my_app.ui' in Qt Designer (e.g., a QMainWindow with a QPushButton named 'myButton').
# 2. Run: pyside6-uic my_app.ui -o ui_my_app.py
#
# ui_my_app.py (Generated file - DO NOT MODIFY MANUALLY)
# from PySide6 import QtCore, QtWidgets
# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.myButton = QtWidgets.QPushButton(self.centralwidget)
#         self.myButton.setObjectName("myButton")
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)
#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("MainWindow", "My App"))
#         self.myButton.setText(_translate("MainWindow", "Click Me"))

# main.py (Controller class)
from PySide6.QtWidgets import QApplication, QMainWindow
from ui_my_app import Ui_MainWindow # Import the generated UI class
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self) # Initialize the UI from the generated class
        self.setWindowTitle("Good UI Design - Composed") # Override title if needed

        # Connect signals AFTER setupUi
        self.ui.myButton.clicked.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        print("Button clicked from composed UI!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

## 2. Modern Controls & QML

**Principle:** Embrace modern UI/UX with Qt Quick Controls 2 for fluid, material-style components. Use QML for declarative UI, and Python for business logic.

**Rule:** Prefer Qt Quick Controls 2 for new UIs. Only create custom QML controls when built-in options are insufficient. Keep QML focused on UI declaration; expose data and complex logic from Python via `QObject` properties and slots.

❌ **BAD:** Mixing complex business logic directly into QML, or using QWidgets for highly dynamic/animated interfaces where QML excels.
```qml
// main.qml (Bad: Complex logic in QML)
import QtQuick
import QtQuick.Controls

ApplicationWindow {
    width: 640; height: 480; visible: true
    title: "Bad QML - Logic in UI"

    TextField { id: inputField; text: "10" }
    Button {
        text: "Calculate Factorial"
        onClicked: {
            // Bad: Complex calculation directly in QML
            let n = parseInt(inputField.text);
            let result = 1;
            for (let i = 2; i <= n; i++) {
                result *= i;
            }
            resultLabel.text = "Factorial: " + result;
        }
    }
    Label { id: resultLabel; text: "Result: " }
}
```

✅ **GOOD:** QML for UI, Python for logic.
```python
# backend.py (Python backend for QML)
from PySide6.QtCore import QObject, Property, Signal, Slot
import math

class Backend(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_value = "10"
        self._result_value = ""

    inputValueChanged = Signal()
    resultValueChanged = Signal()

    @Property(str, notify=inputValueChanged)
    def inputValue(self) -> str:
        return self._input_value

    @inputValue.setter
    def inputValue(self, value: str):
        if self._input_value != value:
            self._input_value = value
            self.inputValueChanged.emit()

    @Property(str, notify=resultValueChanged)
    def resultValue(self) -> str:
        return self._result_value

    @resultValue.setter
    def resultValue(self, value: str):
        if self._result_value != value:
            self._result_value = value
            self.resultValueChanged.emit()

    @Slot()
    def calculateFactorial(self) -> None:
        try:
            n = int(self.inputValue)
            if n < 0:
                self.resultValue = "Error: Negative input"
            else:
                self.resultValue = f"Factorial: {math.factorial(n)}"
        except ValueError:
            self.resultValue = "Error: Invalid number"

# main.py (QML Application entry point)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
import sys
from backend import Backend

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend) # Expose Python object to QML

    engine.load("main.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())

# main.qml (QML UI - uses backend)
import QtQuick
import QtQuick.Controls

ApplicationWindow {
    width: 640; height: 480; visible: true
    title: "Good QML - UI Only"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20

        TextField {
            Layout.fillWidth: true
            placeholderText: "Enter a number"
            text: backend.inputValue // Bind to Python property
            onTextChanged: backend.inputValue = text
        }

        Button {
            Layout.fillWidth: true
            text: "Calculate Factorial"
            onClicked: backend.calculateFactorial() // Call Python slot
        }

        Label {
            Layout.fillWidth: true
            text: backend.resultValue // Bind to Python property
        }
    }
}
```

## 3. Type Safety & Linting

**Principle:** Leverage PySide6's robust type hints for early error detection and improved code readability.

**Rule:** Always use type hints for all PySide6 properties, signals, slots, and method parameters. Enforce PEP 8 naming conventions (snake_case for Python functions/variables, CamelCase for Qt classes/methods) using `black` for formatting and `mypy`/`pylint` for static analysis in CI.

❌ **BAD:** Untyped code, inconsistent naming.
```python
# Bad: No type hints, inconsistent naming
class MyWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.my_button = QPushButton("Click")
        self.my_button.clicked.connect(self.handle_click)

    def handle_click(self): # Should be _handle_click or handle_click_event
        print("Clicked")

def AnotherFunction(arg): # Should be another_function
    pass
```

✅ **GOOD:** Fully typed, PEP 8 compliant, clear intent.
```python
# Good: Type hints, PEP 8 compliant
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
from PySide6.QtCore import Slot

class MyWidget(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.my_button: QPushButton = QPushButton("Click Me")
        self.setCentralWidget(self.my_button)
        self.my_button.clicked.connect(self._handle_click) # Private method convention

    @Slot()
    def _handle_click(self) -> None:
        """Handles the button click event."""
        print("Button was clicked!")

def another_helper_function(data: str) -> bool:
    """A helper function with type hints."""
    return len(data) > 0
```

## 4. Styling & Theming

**Principle:** Achieve a consistent, modern UI aesthetic across your application.

**Rule:** Use `QtVSCodeStyle` for applying modern, VS Code-inspired themes. Always set the `Qt.ApplicationAttribute.AA_UseHighDpiPixmaps` attribute for crisp SVG icons on high-DPI displays.

❌ **BAD:** Default Qt styling, blurry icons on high-DPI screens.
```python
# Bad: Default Qt style, no high-DPI fix
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

app = QApplication(sys.argv)
# No stylesheet, no high-DPI attribute
main_win = QMainWindow()
push_button = QPushButton("Unstyled Button")
main_win.setCentralWidget(push_button)
main_win.show()
sys.exit(app.exec())
```

✅ **GOOD:** Consistent theme, high-DPI support.
```python
# Good: QtVSCodeStyle, high-DPI fix
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from PySide6.QtCore import Qt
import qtvscodestyle as qtvsc

app = QApplication(sys.argv)
app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps) # Essential for crisp SVGs

# Load a dark VS Code theme
stylesheet = qtvsc.load_stylesheet(qtvsc.Theme.DARK_VS)
app.setStyleSheet(stylesheet)

main_win = QMainWindow()
push_button = QPushButton("Styled Button")
main_win.setCentralWidget(push_button)
main_win.show()
sys.exit(app.exec())
```

## 5. Signal/Slot Hygiene

**Principle:** Maintain clean, readable, and robust signal/slot connections.

**Rule:** Use the modern `signal.connect(slot)` syntax. Avoid lambda-heavy connections inside loops; use `functools.partial` or dedicated methods for passing arguments. Encapsulate complex slot logic in separate, well-named methods.

❌ **BAD:** Old-style connections, lambdas in loops leading to closure issues.
```python
# Bad: Old-style connect, lambda in loop (i will always be 4)
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        for i in range(5):
            button = QPushButton(f"Button {i}")
            layout.addWidget(button)
            # button.clicked.connect(lambda: self.handle_button(i)) # 'i' is captured by reference!
            # self.connect(button, SIGNAL("clicked()"), self.handle_button_old) # Old style
```

✅ **GOOD:** New-style connections, proper argument handling with `functools.partial`.
```python
# Good: New-style, proper argument handling
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Slot
from functools import partial
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        for i in range(5):
            button = QPushButton(f"Button {i}")
            layout.addWidget(button)
            # Use functools.partial for arguments in loops to capture 'i' correctly
            button.clicked.connect(partial(self._handle_button_with_arg, i))

        self.another_button = QPushButton("Simple Action")
        layout.addWidget(self.another_button)
        self.another_button.clicked.connect(self._handle_simple_action)

    @Slot(int)
    def _handle_button_with_arg(self, index: int) -> None:
        print(f"Button {index} clicked!")
        # Delegate complex logic to other methods if needed

    @Slot()
    def _handle_simple_action(self) -> None:
        print("Simple action triggered!")
        self._perform_complex_sub_action()

    def _perform_complex_sub_action(self) -> None:
        # ... actual complex logic ...
        print("Complex sub-action completed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
```

## 6. Performance & Concurrency

**Principle:** Maintain a responsive UI by offloading long-running tasks from the main thread.

**Rule:** Never perform blocking I/O (e.g., network requests, file operations) or