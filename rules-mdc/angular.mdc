---
description: Provides definitive guidelines for modern Angular development (2025), focusing on best practices for code structure, component architecture, state management with signals, performance, and type safety.
globs: **/*.{ts,html}
---
# Angular Best Practices

This guide outlines the definitive best practices for Angular development, ensuring consistency, maintainability, and optimal performance. Adhere to these rules strictly.

## 1. Code Organization and Structure

### 1.1 One Concept Per File (SRP)

**Rule:** Each file must contain a single Angular concept (e.g., one component, one directive, one pipe, one service, one interface).

**Explanation:** This enforces the Single Responsibility Principle (SRP), improving modularity, readability, and testability. Exceptions are rare and only for tightly coupled, non-exported elements.

**❌ BAD:**
```typescript
// user.model-and-service.ts
export interface User { id: string; name: string; }

@Injectable({ providedIn: 'root' })
export class UserService {
  // ... fetches users
}
```

**✅ GOOD:**
```typescript
// user.model.ts
export interface User { id: string; name: string; }

// user.service.ts
@Injectable({ providedIn: 'root' })
export class UserService {
  // ... fetches users
}
```

### 1.2 Consistent Naming Conventions

**Rule:** Follow kebab-case for file names (`feature-name.type.ts`) and PascalCase for class names. Use standard type suffixes.

**Explanation:** Consistent naming makes files easy to locate and understand at a glance.

**❌ BAD:**
```typescript
// MyUserList.component.ts
export class UserListComp { /* ... */ }

// authservice.ts
export class AuthenticationService { /* ... */ }
```

**✅ GOOD:**
```typescript
// user-list.component.ts
export class UserListComponent { /* ... */ }

// auth.service.ts
export class AuthService { /* ... */ }
```

## 2. Component Architecture

### 2.1 Standalone Components by Default

**Rule:** Always create standalone components, directives, and pipes. Avoid NgModules for new features unless strictly necessary for legacy integration or specific library patterns.

**Explanation:** Standalone components simplify the Angular mental model, reduce boilerplate, and improve tree-shaking.

**❌ BAD:**
```typescript
// app.module.ts
@NgModule({
  declarations: [MyComponent],
  imports: [CommonModule],
  // ...
})
export class AppModule { }

// my.component.ts
@Component({
  selector: 'app-my',
  template: `<p>My Component</p>`,
  styleUrls: ['./my.component.css'],
})
export class MyComponent { }
```

**✅ GOOD:**
```typescript
// my.component.ts
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common'; // Import dependencies directly

@Component({
  selector: 'app-my',
  standalone: true, // Always standalone
  imports: [CommonModule],
  template: `<p>My Component</p>`,
  styleUrls: ['./my.component.css'],
})
export class MyComponent { }
```

### 2.2 Prefer `inject` for Dependency Injection

**Rule:** Use the `inject` function for all dependency injection, especially in field initializers. Avoid constructor injection unless required for specific lifecycle hooks or inheritance patterns.

**Explanation:** `inject` offers greater flexibility, better tree-shaking, and allows injection outside of constructors.

**❌ BAD:**
```typescript
import { MyService } from './my.service';

@Component({...})
export class MyComponent {
  private myService: MyService;

  constructor(myService: MyService) {
    this.myService = myService;
  }
}
```

**✅ GOOD:**
```typescript
import { inject } from '@angular/core';
import { MyService } from './my.service';

@Component({...})
export class MyComponent {
  // Use a private readonly field for injected services
  private readonly myService = inject(MyService);

  // You can also inject in functions if needed (e.g., providers)
  // readonly myOtherService = someFactoryFn(inject(AnotherService));
}
```

## 3. State Management

### 3.1 Prioritize Signals for Component State

**Rule:** Use Angular Signals for all local component state and simple derived state. Reserve RxJS for complex asynchronous data streams, side effects, and inter-component communication.

**Explanation:** Signals provide a reactive, performant, and simpler way to manage state, optimizing change detection automatically.

**❌ BAD:**
```typescript
// Using plain properties and relying on Zone.js for change detection
@Component({...})
export class MyComponent {
  count = 0;
  items: string[] = [];

  addItem(item: string) {
    this.items.push(item); // Direct mutation, less reactive
  }
}
```

**✅ GOOD:**
```typescript
import { Component, signal, computed } from '@angular/core';

@Component({...})
export class MyComponent {
  readonly count = signal(0); // Primitive signal
  readonly items = signal<string[]>([]); // Array signal

  // Derived state with computed
  readonly hasItems = computed(() => this.items().length > 0);

  increment() {
    this.count.update(value => value + 1);
  }

  addItem(item: string) {
    this.items.update(currentItems => [...currentItems, item]); // Immutable update
  }
}
```

### 3.2 RxJS for Asynchronous Streams and Effects

**Rule:** Leverage RxJS for handling HTTP requests, complex event streams, and orchestrating side effects. Always manage subscriptions to prevent memory leaks.

**Explanation:** RxJS is powerful for reactive programming with asynchronous data. Use `toSignal` or `takeUntilDestroyed` for automatic unsubscription.

**❌ BAD:**
```typescript
// Manual subscription management
ngOnInit() {
  this.userService.getUsers().subscribe(users => {
    this.users = users;
  });
}

ngOnDestroy() {
  // Forgot to unsubscribe, potential memory leak
}
```

**✅ GOOD:**
```typescript
import { Component, inject, DestroyRef } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop'; // For converting RxJS to Signal
import { UserService } from './user.service';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { tap } from 'rxjs';

@Component({...})
export class MyComponent {
  private readonly userService = inject(UserService);
  private readonly destroyRef = inject(DestroyRef);

  // Convert an RxJS Observable to a Signal, automatically unsubscribes
  readonly users = toSignal(this.userService.getUsers(), { initialValue: [] });

  // For effects or more complex streams, manage with takeUntilDestroyed
  constructor() {
    this.userService.someAction$
      .pipe(
        tap(action => console.log('Action:', action)),
        takeUntilDestroyed(this.destroyRef) // Automatic unsubscription
      )
      .subscribe();
  }
}
```

## 4. Performance Considerations

### 4.1 Use Built-in Control Flow

**Rule:** Discontinue the use of structural directives (`*ngIf`, `*ngFor`, `*ngSwitch`). Always use Angular's new built-in control flow syntax.

**Explanation:** The new control flow (`@if`, `@for`, `@switch`) offers improved performance, better type checking, and a more intuitive syntax.

**❌ BAD:**
```html
<!-- Using *ngIf -->
<div *ngIf="isLoading">Loading...</div>

<!-- Using *ngFor -->
<li *ngFor="let item of items; let i = index">{{ item.name }}</li>
```

**✅ GOOD:**
```html
<!-- Using @if -->
@if (isLoading) {
  <div>Loading...</div>
} @else {
  <div>Data loaded!</div>
}

<!-- Using @for with track -->
@for (item of items; track item.id) {
  <li>{{ item.name }}</li>
} @empty {
  <p>No items found.</p>
}
```

### 4.2 Always Use `track` Key in `@for`

**Rule:** Always provide a `track` key in `@for` loops. Use a unique, primitive property of the object (e.g., `item.id`). If iterating over primitives, use the value directly.

**Explanation:** The `track` key helps Angular optimize rendering by identifying unique items, preventing unnecessary DOM re-renders and improving performance.

**❌ BAD:**
```html
<!-- Angular will re-render all items if the array reference changes -->
@for (user of users) {
  <li>{{ user.name }}</li>
}
```

**✅ GOOD:**
```html
<!-- Optimized re-rendering based on user.id -->
@for (user of users; track user.id) {
  <li>{{ user.name }}</li>
}

<!-- For primitive arrays, track the value itself -->
@for (name of names; track name) {
  <li>{{ name }}</li>
}
```

## 5. Type Safety

### 5.1 Define Interfaces for Data Structures

**Rule:** Always define TypeScript interfaces or types for all data structures, especially those received from APIs or used in models.

**Explanation:** Interfaces provide strong type checking, improve code readability, and catch errors early.

**❌ BAD:**
```typescript
// Loosely typed data
fetchUsers().subscribe((data: any) => {
  console.log(data.users[0].firstName); // Prone to runtime errors
});
```

**✅ GOOD:**
```typescript
export interface User {
  id: string;
  firstName: string;
  lastName: string;
  email: string;
}

fetchUsers().subscribe((data: User[]) => {
  console.log(data[0].firstName); // Type-safe access
});
```

### 5.2 Default to `readonly` Properties

**Rule:** Declare component properties and injected services as `readonly` where their values are not intended to be reassigned after initialization.

**Explanation:** `readonly` encourages immutability, prevents accidental reassignments, and makes state flow more predictable.

**❌ BAD:**
```typescript
@Component({...})
export class MyComponent {
  title = 'My App'; // Can be reassigned anywhere
  private myService = inject(MyService); // Can be reassigned
}
```

**✅ GOOD:**
```typescript
@Component({...})
export class MyComponent {
  readonly title = 'My App'; // Immutable after initialization
  private readonly myService = inject(MyService); // Immutable reference to service
}
```

## 6. Accessibility (A11y)

### 6.1 Leverage Angular Aria and CDK

**Rule:** Integrate Angular Aria and the Component Dev Kit (CDK) for building accessible UI components. Always provide appropriate ARIA attributes and semantic HTML.

**Explanation:** Angular's built-in tools help ensure your applications are usable by everyone, including those with disabilities.

**❌ BAD:**
```html
<!-- Custom button without proper ARIA roles -->
<div (click)="doSomething()">Click Me</div>
```

**✅ GOOD:**
```html
<!-- Semantic button with ARIA attributes for accessibility -->
<button type="button" (click)="doSomething()" aria-label="Perform action">Click Me</button>

<!-- Using CDK for accessible interactions, e.g., cdkMenu -->
<button cdkMenuTrigger>Open Menu</button>
<div cdkMenu>
  <button cdkMenuItem>Item 1</button>
</div>
```