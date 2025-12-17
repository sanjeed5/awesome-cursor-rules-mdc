---
description: Definitive guidelines for writing clean, maintainable, and performant Emacs Lisp code and configurations, emphasizing modern best practices, robust organization, and effective tooling.
globs: **/*
---
# emacs Best Practices

This guide outlines the definitive best practices for Emacs Lisp development and configuration. Adhere to these rules to ensure your code is maintainable, performant, and integrates seamlessly with the Emacs ecosystem.

## 1. Code Organization and Structure

### 1.1 File Headers

Every Emacs Lisp file **must** start with a standard header, including `lexical-binding` and a clear description.

❌ BAD:
```emacs-lisp
;; my-feature.el
(defun my-feature-do-something () ...)
```

✅ GOOD:
```emacs-lisp
;;; my-feature.el --- A short description of my-feature -*- lexical-binding: t; -*-
;;
;; Copyright (C) 2025 Your Name
;; Author: Your Name <your.email@example.com>
;; Keywords: convenience, tools
;;
;;; Commentary:
;; This library provides functions for doing X, Y, and Z.
;;
;;; Code:

(require 'cl-lib) ; Example dependency

(provide 'my-feature)
;;; my-feature.el ends here
```

### 1.2 Naming Conventions

All global symbols (functions, variables, constants) **must** be prefixed with a short, hyphen-separated package name. Use a double hyphen (`--`) for internal, non-public symbols.

❌ BAD:
```emacs-lisp
(defvar foo-list nil "A list of foos.")
(defun do-something () ...)
```

✅ GOOD:
```emacs-lisp
(defvar myproj-foo-list nil "A list of foos for My Project.")
(defun myproj-do-something (arg) ...)
(defun myproj--internal-helper (arg) ...) ; Internal function
```

### 1.3 `init.el` Configuration

Your `init.el` **must** be a single file, primarily using `use-package`. If you prefer a literate Org file, **always** tangle it to `init.el`; **never** use `org-babel-load-file`.

❌ BAD:
```emacs-lisp
;; init.el
(org-babel-load-file "~/.emacs.d/init.org")
(load-file "~/.emacs.d/my-custom-settings.el")
(setq global-setting t) ; Scattered settings
```

✅ GOOD:
```emacs-lisp
;; init.el (or tangled from init.org)
;; -*- lexical-binding: t; -*-

;; Package manager setup (e.g., straight.el or elpaca)
(setq package-enable-at-startup nil)
(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))
(eval-when-compile (require 'use-package))
(require 'use-package-ensure) ; For :ensure t

;; Global Emacs settings (use the 'emacs' pseudo-feature)
(use-package emacs
  :init
  (setq custom-file (expand-file-name "custom.el" user-emacs-directory))
  (load custom-file 'noerror 'nomessage)
  (setq-default fill-column 80
                indent-tabs-mode nil
                tab-width 2)
  :config
  (global-display-line-numbers-mode))

;; Example package configuration
(use-package my-package
  :ensure t
  :defer t ; Defer loading until needed
  :bind (("C-c m" . my-package-command))
  :config
  (setq my-package-setting t))
```

### 1.4 Docstrings and Comments

Every public function and variable **must** have a docstring. The first line **must** be a complete sentence. Use comments to explain complex logic, not obvious code.

❌ BAD:
```emacs-lisp
(defun myproj-add (a b) (+ a b)) ; adds two numbers
```

✅ GOOD:
```emacs-lisp
(defun myproj-add (a b)
  "Return the sum of A and B.
This function performs basic arithmetic addition."
  (+ a b))

(defvar myproj-cache-size 100
  "Maximum number of items to store in the project cache.")

;; This complex regex matches URLs with optional query parameters.
;; It ensures we capture the base URL separately.
(let ((url-regex "^\\(https?://[^/?#]+\\)\\(?:[/?#].*\\)?$"))
  ...)
```

## 2. Common Patterns and Anti-patterns

### 2.1 Lexical Binding

**Always** enable lexical binding. It prevents subtle bugs related to dynamic scope and allows for proper closures. Add `-*- lexical-binding: t; -*-` to your file header.

### 2.2 Variable Declarations

Use `let*` for sequential binding and align variables for readability. Avoid redundant `nil` initializations.

❌ BAD:
```emacs-lisp
(let ((foo '()) (bar nil) baz))
```

✅ GOOD:
```emacs-lisp
(let* ((myproj-foo-list nil) ; A list of processed items
       (myproj-bar-flag nil) ; A boolean flag indicating status
       myproj-temp-var)      ; A scratch variable for intermediate calculations
  ...)
```

### 2.3 Keybinding Conventions

**Never** bind `C-c <letter>` (e.g., `C-c a`); these are reserved for users. Use `C-c <non-letter>` for major mode commands. **Never** bind `C-h` after any prefix.

❌ BAD:
```emacs-lisp
(global-set-key (kbd "C-c a") 'myproj-action)
(define-key my-mode-map (kbd "C-c h") 'myproj-help)
```

✅ GOOD:
```emacs-lisp
(global-set-key (kbd "C-c @") 'myproj-global-action) ; C-c followed by a non-letter
(define-key my-mode-map (kbd "C-c ,") 'myproj-mode-action)
```

### 2.4 Do Not Redefine Emacs Primitives

**Never** redefine or alias standard Emacs Lisp primitives (e.g., `copy-list`, `cadr`). This leads to unpredictable behavior and breaks other packages.

❌ BAD:
```emacs-lisp
(defun copy-list (list) (cl-copy-list list)) ; Redefining a built-in
```

✅ GOOD:
```emacs-lisp
;; Use the standard primitive directly or define your own with a proper prefix.
(myproj-custom-copy-list my-list)
```

## 3. Performance Considerations

### 3.1 Byte Compilation

**Always** byte-compile your Emacs Lisp files. Use `auto-compile` to automate this on save.

❌ BAD:
```emacs-lisp
;; Only distribute .el files
```

✅ GOOD:
```emacs-lisp
;; Ensure .elc files are generated and kept up-to-date
(use-package auto-compile
  :ensure t
  :config
  (auto-compile-mode 1))
```

### 3.2 `init.el` Load Time

Keep your `init.el` lean. Defer package loading with `use-package`'s `:defer`, `:demand`, `:bind`, `:hook` keywords.

❌ BAD:
```emacs-lisp
(require 'heavy-package) ; Loads immediately
```

✅ GOOD:
```emacs-lisp
(use-package heavy-package
  :ensure t
  :defer t ; Only load when 'heavy-package' functionality is called
  :config
  (setq heavy-package-setting t))
```

## 4. Common Pitfalls and Gotchas

### 4.1 Mark Management

**Never** set the mark in your Lisp code unless you are writing a user-facing command specifically designed to manipulate the mark ring. Functions like `beginning-of-buffer` or `replace-string` set the mark; use lower-level alternatives for programmatic movement/replacement.

❌ BAD:
```emacs-lisp
(defun myproj-process-buffer ()
  (interactive)
  (beginning-of-buffer) ; Sets the mark
  (replace-string "old" "new")) ; Sets the mark
```

✅ GOOD:
```emacs-lisp
(defun myproj-process-buffer ()
  "Process the current buffer, replacing 'old' with 'new'."
  (interactive)
  (save-excursion ; Preserve point and mark
    (goto-char (point-min))
    (while (re-search-forward "old" nil t)
      (replace-match "new"))))
```

### 4.2 Buffer Navigation

**Always** use `forward-line` for programmatic line movement. `next-line` and `previous-line` are unreliable in Lisp code as they depend on visual line boundaries.

❌ BAD:
```emacs-lisp
(next-line)
```

✅ GOOD:
```emacs-lisp
(forward-line 1)
```

### 4.3 Messaging and Errors

Use `message` for user feedback in the echo area. Use `error` or `signal` for unrecoverable conditions.

❌ BAD:
```emacs-lisp
(princ "Something happened!")
(if (not condition) (throw 'error "Failed!"))
```

✅ GOOD:
```emacs-lisp
(message "My project: Something happened!")
(unless condition
  (error "My project: Condition not met, cannot proceed."))
```

## 5. Testing Approaches

### 5.1 Unit Testing with ERT

**Always** write unit tests for your Emacs Lisp libraries using the built-in Emacs Lisp Regression Test (ERT) framework.

```emacs-lisp
;; my-feature-test.el
;;; my-feature-test.el --- Tests for my-feature -*- lexical-binding: t; -*-
;;; Code:

(require 'ert)
(require 'my-feature)

(ert-deftest myproj-add-basic-test ()
  "Test basic addition with `myproj-add`."
  (should (= (myproj-add 1 2) 3)))

(provide 'my-feature-test)
;;; my-feature-test.el ends here
```

### 5.2 Linting and Static Analysis

**Always** use `flycheck` with `flycheck-package` to catch common errors, naming violations, and documentation issues early.

```emacs-lisp
;; Ensure flycheck is enabled in your init.el
(use-package flycheck
  :ensure t
  :init (global-flycheck-mode))

(use-package flycheck-package
  :ensure t
  :after flycheck
  :config
  (add-hook 'flycheck-mode-hook 'flycheck-package-setup))
```

### 5.3 Interactive Debugging

Leverage Emacs's built-in `edebug` for interactive debugging. Use `C-u C-M-x` on a `defun` to instrument a function, then call it to step through.

```emacs-lisp
;; Place point on 'defun', then C-u C-M-x to instrument
(defun myproj-debug-example (arg)
  "An example function to debug."
  (let ((result (* arg 2)))
    (message "Intermediate result: %s" result)
    (+ result 10)))

(myproj-debug-example 5) ; Call this after instrumenting to start edebug
```