---
description: Provides definitive guidelines for building scalable, performant, and maintainable React applications using Material-UI (MUI), focusing on modern best practices.
globs: **/*
---
# Material-UI (MUI) Best Practices

This guide outlines the definitive best practices for developing React applications with Material-UI (MUI). Adhere to these standards to ensure consistent, performant, and maintainable code across our projects.

## 1. Code Organization & Structure

Organize code by feature, not by type. Components should be small, focused, and self-contained.

### 1.1 Directory Structure
Adopt a feature-oriented layout.
```
src/
  components/           # Reusable, generic UI components (e.g., Button, Modal)
    Button/
      Button.jsx
      Button.styled.js  # Styled components for Button
      Button.test.jsx
      index.js          # Exports Button
  features/             # Feature-specific components and logic (e.g., UserProfile, ProductList)
    Auth/
      LoginPage.jsx
      AuthForm.jsx
      AuthService.js
      AuthContext.js
  theme/                # Centralized MUI theme configuration
    index.js
    palette.js
    typography.js
  utils/                # General utility functions
    api.js
    helpers.js
  App.jsx               # Main application component
  index.jsx             # Entry point
```

### 1.2 Naming Conventions
-   **Components**: `PascalCase` (e.g., `MyComponent.jsx`).
-   **Styled Components**: `PascalCase` for the component, `.styled.js` suffix (e.g., `MyComponent.styled.js`).
-   **CSS Modules**: `kebab-case` or `camelCase` (e.g., `my-component.module.css`).
-   **Tests**: `.test.jsx` suffix (e.g., `MyComponent.test.jsx`).
-   **Indexes**: `index.js` should export the main entity of its parent folder.

## 2. Theming

Centralize all design tokens and theme customizations. Always use `createTheme` and `ThemeProvider`.

### 2.1 Centralized Theme Configuration
Define your theme in `src/theme/index.js` and wrap your application with `ThemeProvider`. This is where Material Design 3 tokens should be configured.

```jsx
// src/theme/index.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light', // or 'dark'
    primary: {
      main: '#006B5F', // Material Design 3 primary color example
      light: '#4DA89F',
      dark: '#003730',
    },
    secondary: {
      main: '#6A5F00', // Material Design 3 secondary color example
    },
    // ... other palette colors
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
    h1: { fontSize: '2.5rem', fontWeight: 700 },
    body1: { fontSize: '1rem', lineHeight: 1.5 },
  },
  components: {
    MuiButton: {
      defaultProps: {
        disableElevation: true, // Consistent button style
      },
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none', // Prefer sentence case for buttons
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
        },
      },
    },
  },
});

export default theme;

// src/App.jsx
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline'; // Resets browser styles
import theme from './theme';
import MyPage from './features/MyPage';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline /> {/* Always include CssBaseline */}
      <MyPage />
    </ThemeProvider>
  );
}
export default App;
```

### ❌ BAD: Mutating Theme Directly
Never modify the theme object outside of `createTheme`.

```jsx
// ❌ BAD: Directly modifying theme properties after creation
import theme from './theme';
theme.palette.primary.main = '#FF0000'; // This will cause issues and is not reactive
```

## 3. Styling

Prioritize the `sx` prop for simple, one-off overrides and the `styled` API for reusable, complex styles.

### 3.1 `sx` Prop for Quick Overrides
Use `sx` for component-specific, ad-hoc styling that leverages theme tokens and provides responsive capabilities.

```jsx
import Button from '@mui/material/Button';
import Box from '@mui/material/Box';

function MyComponent() {
  return (
    <Box sx={{ p: { xs: 2, sm: 3 }, display: 'flex', gap: 2 }}>
      <Button
        variant="contained"
        sx={{
          mt: 2, // margin-top from theme spacing
          bgcolor: 'primary.dark', // background color from theme palette
          '&:hover': {
            bgcolor: 'primary.main',
          },
        }}
      >
        Click Me
      </Button>
    </Box>
  );
}
```

### 3.2 `styled` API for Reusable Styles
For complex, reusable styles, create dedicated styled components. This ensures consistency and maintainability.

```jsx
// src/components/MyButton/MyButton.styled.js
import { styled } from '@mui/material/styles';
import Button from '@mui/material/Button';

export const StyledButton = styled(Button)(({ theme }) => ({
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(1.5, 3),
  textTransform: 'none',
  fontSize: '1rem',
  [theme.breakpoints.down('sm')]: { // Responsive styling within styled component
    padding: theme.spacing(1, 2),
    fontSize: '0.875rem',
  },
}));

// src/components/MyButton/MyButton.jsx
import { StyledButton } from './MyButton.styled';

function MyButton({ children, ...props }) {
  return <StyledButton {...props}>{children}</StyledButton>;
}
export default MyButton;
```

### ❌ BAD: Inline Styles & Deprecated JSS
Avoid inline styles as they lack theming support and are hard to maintain. Do not use `@mui/styles` (JSS) as it is deprecated.

```jsx
// ❌ BAD: Inline styles
<Button style={{ marginTop: '16px', backgroundColor: '#1976d2' }}>
  Submit
</Button>

// ❌ BAD: Using deprecated JSS (e.g., makeStyles)
import { makeStyles } from '@mui/styles'; // This is deprecated!
const useStyles = makeStyles({ /* ... */ });
```

## 4. Component Architecture

Favor functional components, hooks, and composition. Keep components small and focused.

### 4.1 Functional Components & Hooks
Always use functional components with React Hooks.

```jsx
// ✅ GOOD
import React, { useState, useEffect } from 'react';
import { Box, Typography } from '@mui/material';

function UserProfile({ userId }) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const fetchUser = async () => {
      // Simulate API call
      const response = await new Promise(resolve => setTimeout(() => resolve({ id: userId, name: `User ${userId}` }), 500));
      setUser(response);
    };
    fetchUser();
  }, [userId]);

  if (!user) return <Typography>Loading user...</Typography>;

  return (
    <Box>
      <Typography variant="h4">{user.name}</Typography>
      {/* ... other user details */}
    </Box>
  );
}

// ❌ BAD: Class components
class UserProfile extends React.Component {
  // ... use componentDidMount, setState, etc.
}
```

### 4.2 Small, Purpose-Driven Components
Break down complex UIs into smaller, reusable components. This improves readability, testability, and reusability.

```jsx
// ❌ BAD: Large, monolithic component
function ProductCard({ product }) {
  return (
    <Card>
      <CardMedia component="img" image={product.image} alt={product.name} />
      <CardContent>
        <Typography variant="h5">{product.name}</Typography>
        <Typography variant="body2">{product.description}</Typography>
        <Typography variant="h6">${product.price}</Typography>
        <Rating value={product.rating} readOnly />
        <Button onClick={() => console.log('Add to cart')}>Add to Cart</Button>
        {/* Many other details like stock, seller info, etc. */}
      </CardContent>
    </Card>
  );
}

// ✅ GOOD: Composed components
function ProductImage({ src, alt }) {
  return <CardMedia component="img" image={src} alt={alt} sx={{ height: 140 }} />;
}
function ProductDetails({ name, description, price, rating }) {
  return (
    <>
      <Typography variant="h5">{name}</Typography>
      <Typography variant="body2" color="text.secondary">{description}</Typography>
      <Typography variant="h6" mt={1}>${price}</Typography>
      <Rating value={rating} readOnly size="small" />
    </>
  );
}
function AddToCartButton({ onClick }) {
  return <Button variant="contained" onClick={onClick} sx={{ mt: 2 }}>Add to Cart</Button>;
}

function ProductCard({ product }) {
  return (
    <Card sx={{ maxWidth: 345 }}>
      <ProductImage src={product.image} alt={product.name} />
      <CardContent>
        <ProductDetails {...product} />
        <AddToCartButton onClick={() => console.log(`Added ${product.name} to cart`)} />
      </CardContent>
    </Card>
  );
}
```

### 4.3 `@mui/base` for Custom Design Systems
When building a highly custom design system that doesn't adhere to Material Design, use `@mui/base` for unstyled components and hooks. This provides maximum flexibility.

```jsx
// ✅ GOOD: Using @mui/base for a custom unstyled button
import { useButton } from '@mui/base/useButton';
import { styled } from '@mui/system';

const CustomButtonRoot = styled('button')`
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: #3f51b5; /* Custom brand color */
  padding: 12px 24px;
  border-radius: 8px;
  color: white;
  transition: all 150ms ease;
  cursor: pointer;
  border: none;

  &:hover {
    background-color: #303f9f;
  }
  &:active {
    background-color: #283593;
  }
`;

function CustomButton(props) {
  const { children, ...other } = props;
  const { getRootProps } = useButton(other); // Provides accessibility attributes and event handlers
  return <CustomButtonRoot {...getRootProps()}>{children}</CustomButtonRoot>;
}
```

## 5. Hooks Best Practices

Leverage React Hooks for state, effects, and performance optimizations.

### 5.1 `useMemo` & `useCallback` for Performance
Memoize expensive computations and function references to prevent unnecessary re-renders of child components. Only use when profiling indicates a performance bottleneck.

```jsx
import React, { useState, useMemo, useCallback } from 'react';
import { List, ListItem, ListItemText } from '@mui/material';

function ItemList({ items, filterText }) {
  const [count, setCount] = useState(0);

  // ✅ GOOD: Memoize filteredItems to re-calculate only when items or filterText changes
  const filteredItems = useMemo(() => {
    console.log('Filtering items...');
    return items.filter(item => item.name.toLowerCase().includes(filterText.toLowerCase()));
  }, [items, filterText]);

  // ✅ GOOD: Memoize handler to prevent unnecessary re-renders of child components if passed as prop
  const handleItemClick = useCallback((itemId) => {
    console.log(`Item ${itemId} clicked`);
  }, []); // Empty dependency array if it doesn't use props/state from its closure

  return (
    <>
      <button onClick={() => setCount(c => c + 1)}>Increment: {count}</button>
      <List>
        {filteredItems.map(item => (
          <ListItem key={item.id} onClick={() => handleItemClick(item.id)}>
            <ListItemText primary={item.name} />
          </ListItem>
        ))}
      </List>
    </>
  );
}
```

### 5.2 React 19 `useRef` Compatibility
Be aware of React 19's `useRef` changes. For `forwardRef` components, ensure `ref` is handled correctly. MUI components handle this internally, but for custom `forwardRef` implementations, consider a shim for backward compatibility if supporting React < 19.

```jsx
// ✅ GOOD: React 19 compatible forwardRef shim (adapted from MUI X migration strategy)
// This ensures `ref` is always present in props for type safety and stability.
import React from 'react';

// Determine React major version for conditional logic
const reactMajor = parseInt(React.version.split('.')[0], 10);

export const forwardRef = <T, P = {}>(
  render: React.ForwardRefRenderFunction<T, P & { ref: React.Ref<T> }>,
) => {
  if (reactMajor >= 19) {
    // In React 19, ref is passed as a prop, and forwardRef is often not strictly needed
    const Component = (props: any) => render(props, props.ref ?? null);
    Component.displayName = render.displayName ?? render.name;
    return Component as React.ForwardRefExoticComponent<P>;
  }
  // For React < 19, use standard React.forwardRef
  return React.forwardRef(
    render as React.ForwardRefRenderFunction<T, React.PropsWithoutRef<P>>,
  );
};

// Usage with the shim:
const MyForwardedComponent = forwardRef((props, ref) => {
  // Ensure ref