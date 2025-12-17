---
description: This guide defines definitive best practices for developing with shadcn/ui, ensuring consistent code organization, robust TypeScript, optimal performance, and accessible, maintainable components.
globs: **/*
---
# shadcn Best Practices

This document outlines our team's definitive best practices for developing with `shadcn/ui`. Adhere to these guidelines for all `shadcn` component development and integration.

## 1. Code Organization and Structure

Organize components logically to reflect UI hierarchy and promote discoverability.

**Rule:** Place domain-specific components under `components/<domain>` and reusable UI primitives under `components/ui`. Each component must reside in its own PascalCase file.

❌ BAD:
```
// components/Button.tsx
// components/profile-card.tsx
// components/user-settings/index.tsx (contains multiple components)
```

✅ GOOD:
```
// components/ui/Button.tsx
// components/forms/DatePicker.tsx
// components/layout/Sidebar.tsx

// components/forms/index.ts
export * from "./DatePicker";
export * from "./Input";
```

## 2. Component Architecture

Favor functional components, composition, and explicit prop definitions.

**Rule:** Use functional components with `React.forwardRef` and `asChild` for seamless integration with Radix primitives.

❌ BAD:
```tsx
// No ref forwarding, no asChild
const Button = ({ children, onClick }) => (
  <button onClick={onClick}>{children}</button>
);
```

✅ GOOD:
```tsx
// components/ui/Button.tsx
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils"; // Assuming you have a cn utility

const buttonVariants = cva(
  "inline-flex items-center justify-center rounded-md text-sm font-medium",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
```

## 3. TypeScript and API Design

Enforce strict TypeScript with clear interfaces and robust validation.

**Rule:** Use interfaces for component props. Validate form data with Zod schemas. Avoid `any` and prefer explicit types.

❌ BAD:
```typescript
// Vague props, no validation
type UserFormProps = {
  data: any;
  onSubmit: (values: any) => void;
};
```

✅ GOOD:
```typescript
// components/forms/UserForm.tsx
import { z } from "zod";

export interface UserFormProps {
  initialData?: UserFormData;
  onSubmit: (values: UserFormData) => void;
}

export const userFormSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(2, "Name must be at least 2 characters."),
  email: z.string().email("Invalid email address."),
});

export type UserFormData = z.infer<typeof userFormSchema>;
```

## 4. Theming and Styling

Leverage Tailwind CSS and `class-variance-authority` (CVA) for consistent, maintainable styling.

**Rule:** Define component variants using CVA. Use the `cn` utility for conditional class merging. Centralize Tailwind configuration and design tokens.

❌ BAD:
```tsx
// Inconsistent inline styles or direct class manipulation
<button className={`p-2 ${isActive ? 'bg-blue-500' : 'bg-gray-200'}`}>
```

✅ GOOD:
```tsx
// components/ui/Badge.tsx
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        outline: "text-foreground",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
```

## 5. Common Patterns and Anti-patterns

**Rule:** Use React Hook Form with Zod for all forms. Implement early returns and guard clauses for error handling.

❌ BAD:
```tsx
// Deeply nested logic, manual form state
if (data) {
  // ... many lines
  if (isValid) {
    // ... more lines
  }
}
```

✅ GOOD:
```tsx
// Early return for invalid state
if (!user) {
  return <p>User not found.</p>;
}

// React Hook Form + Zod example
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { userFormSchema, UserFormData } from "./UserForm"; // From section 3

function UserProfileForm({ initialData, onSubmit }: UserFormProps) {
  const form = useForm<UserFormData>({
    resolver: zodResolver(userFormSchema),
    defaultValues: initialData,
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {/* Form fields */}
    </form>
  );
}
```

## 6. Performance Considerations

Optimize for fast initial loads and smooth interactions.

**Rule:** Lazy-load heavy UI sections (e.g., dialogs, data tables) using `React.lazy` or dynamic imports. Memoize expensive components and callbacks.

❌ BAD:
```tsx
// Always loads heavy component
import { BigComplexChart } from "./BigComplexChart";
function Dashboard() {
  return <BigComplexChart data={...} />;
}
```

✅ GOOD:
```tsx
// Lazy load for better initial performance
import React from "react";
const LazyBigComplexChart = React.lazy(() => import("./BigComplexChart"));

function Dashboard() {
  const [showChart, setShowChart] = React.useState(false);

  // Memoize callbacks to prevent unnecessary re-renders
  const handleToggleChart = React.useCallback(() => {
    setShowChart((prev) => !prev);
  }, []);

  return (
    <div>
      <Button onClick={handleToggleChart}>Toggle Chart</Button>
      {showChart && (
        <React.Suspense fallback={<div>Loading chart...</div>}>
          <LazyBigComplexChart data={...} />
        </React.Suspense>
      )}
    </div>
  );
}
```

## 7. Accessibility

Build inclusive UIs by leveraging Radix primitives and ARIA attributes.

**Rule:** Always use `shadcn/ui` components as they are built on Radix UI and provide excellent accessibility out-of-the-box. Ensure custom components correctly pass ARIA attributes and manage focus.

❌ BAD:
```tsx
// Custom button without ARIA attributes or proper semantics
<div role="button" onClick={...}>Click me</div>
```

✅ GOOD:
```tsx
// Leverage shadcn/ui's accessible Button
import { Button } from "@/components/ui/Button";
<Button onClick={() => alert("Action!")}>Perform Action</Button>
```

## 8. Common Pitfalls and Gotchas

Avoid these common mistakes to maintain a scalable and robust codebase.

**Rule:** Never directly modify `shadcn/ui` component files. Instead, extend them with `cn` or wrap them in higher-order components. Avoid `dangerouslySetInnerHTML`.

❌ BAD:
```tsx
// Direct modification of a shadcn component (will be overwritten by CLI updates)
// components/ui/button.tsx (modified directly)
```

```tsx
// Security vulnerability
<div dangerouslySetInnerHTML={{ __html: userProvidedContent }} />
```

✅ GOOD:
```tsx
// Extend with cn for custom styles
import { Button } from "@/components/ui/Button";
<Button className="bg-red-500 hover:bg-red-600">Custom Red Button</Button>
```

```tsx
// Sanitize and render text content safely
import DOMPurify from 'dompurify';

const sanitizedContent = DOMPurify.sanitize(userProvidedContent);
return <div className="prose" dangerouslySetInnerHTML={{ __html: sanitizedContent }} />;
// Or even better, avoid dangerouslySetInnerHTML if possible and render text directly
// return <p>{userProvidedContent}</p>;
```