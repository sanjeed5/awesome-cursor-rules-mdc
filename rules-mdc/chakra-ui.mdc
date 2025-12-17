---
description: This guide provides opinionated, actionable best practices for using Chakra UI effectively, focusing on token-first theming, responsive design, component architecture, and performance for React 18+ applications.
globs: **/*
---
# chakra-ui Best Practices

Chakra UI is our definitive choice for building accessible, design-system-driven React applications. Adhere to these guidelines for a consistent, maintainable, and performant codebase.

## 1. Core Setup & Dependencies

Always ensure correct installation and root provider setup for Chakra UI v2+ (React 18+).

❌ BAD: Incorrect or missing dependencies, old React versions.
```javascript
// package.json (missing emotion/framer-motion)
"dependencies": {
  "@chakra-ui/react": "^2.x.x",
  "react": "^17.0.0" // Chakra v2 requires React 18+
}

// App.jsx (missing ChakraProvider)
function App() {
  return <HomePage />;
}
```

✅ GOOD: Use React 18+ and all required peer dependencies. Wrap your app with `<ChakraProvider>`.
```javascript
// package.json
"dependencies": {
  "@chakra-ui/react": "^2.x.x",
  "@emotion/react": "^11.x.x",
  "@emotion/styled": "^11.x.x",
  "framer-motion": "^6.x.x", // or newer compatible versions
  "react": "^18.x.x",
  "react-dom": "^18.x.x"
}

// src/index.jsx or src/main.jsx
import { ChakraProvider } from '@chakra-ui/react';
import React from 'react';
import ReactDOM from 'react-dom/client'; // Use createRoot for React 18
import App from './App';
import theme from './theme'; // Import your custom theme

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>,
);
```

## 2. Theming: Token-First & Recipes

Centralize your design system. Define semantic tokens and use `defineRecipe` for component variants.

### 2.1. Define Semantic Tokens
Avoid hardcoding values. Define colors, fonts, and spacing once with `defineTokens`.

❌ BAD: Inline hex codes or magic numbers.
```javascript
<Box bg="#0FEE0F" p="16px" />
```

✅ GOOD: Use `defineTokens` and reference them via theme keys.
```javascript
// src/theme/tokens.js
import { defineTokens } from "@chakra-ui/react";
export const tokens = defineTokens({
  colors: {
    primary: { value: "#0FEE0F" },
    secondary: { value: "#EE0F0F" },
  },
  space: {
    'md': { value: '16px' }
  },
});

// src/theme/index.js
import { extendTheme } from '@chakra-ui/react';
import { tokens } from './tokens';
const theme = extendTheme({ tokens });
export default theme;

// YourComponent.jsx
<Box bg="primary" p="md" />
```

### 2.2. Use `defineRecipe` for Component Variants
Encapsulate component styles, especially for variants, using `defineRecipe`.

❌ BAD: Scattering variant styles as props or conditional logic in JSX.
```javascript
const Button = ({ variant, children }) => {
  if (variant === 'primary') {
    return <ChakraButton bg="teal.600" color="white">{children}</ChakraButton>;
  }
  return <ChakraButton bg="gray.200">{children}</ChakraButton>;
};
```

✅ GOOD: Define component recipes in your theme.
```javascript
// src/theme/recipes/button.js
import { defineRecipe } from "@chakra-ui/react";
export const buttonRecipe = defineRecipe({
  base: {
    borderRadius: "md",
    fontWeight: "semibold",
  },
  variants: {
    variant: {
      primary: { bg: "primary", color: "white", _hover: { bg: "primary.700" } },
      secondary: { bg: "gray.200", color: "gray.800", _hover: { bg: "gray.300" } },
    },
  },
  defaultVariants: {
    variant: "secondary",
  },
});

// src/theme/index.js (add to components)
import { extendTheme } from '@chakra-ui/react';
import { buttonRecipe } from './recipes/button';
const theme = extendTheme({
  components: {
    Button: buttonRecipe,
  },
});

// YourComponent.jsx
import { Button } from '@chakra-ui/react';
<Button variant="primary">Click Me</Button>
```

## 3. Styling: Shorthand, Pseudo-Props & `Box`

Leverage Chakra's powerful style props for declarative, responsive styling.

### 3.1. Prefer Shorthand Props
Use concise shorthand for common CSS properties.

❌ BAD: Verbose CSS property names.
```javascript
<Box backgroundColor="red.500" margin="4" padding="16px" width="100%" />
```

✅ GOOD: Use `bg`, `m`, `p`, `w`.
```javascript
<Box bg="red.500" m="4" p="4" w="full" />
```

### 3.2. Use Pseudo Props for Interaction States
Handle hover, focus, active, and disabled states directly in JSX.

❌ BAD: Separate CSS classes or conditional styling.
```javascript
// styles.css
.my-button:hover { background: green; }
// MyButton.jsx
<Button className="my-button" />
```

✅ GOOD: Use `_hover`, `_focus`, `_active`, `_disabled`.
```javascript
<Button bg="tomato" _hover={{ bg: "green.500" }} _focus={{ shadow: "outline" }} />
```

### 3.3. `Box` as the Styling Primitive
Use `Box` for layout and atomic styling. It's the foundation for all Chakra components.

❌ BAD: Using `div` with inline styles or external CSS for basic layout.
```javascript
<div style={{ display: 'flex', padding: '16px' }}>...</div>
```

✅ GOOD: Use `Box` (or `Flex`, `Grid`) with Chakra props.
```javascript
import { Box, Flex } from '@chakra-ui/react';
<Flex p="4">...</Flex>
```

## 4. Component Architecture & Accessibility

Build composable, semantic, and accessible components.

### 4.1. Composition over Wrapper Divs
Compose Chakra components directly. Avoid unnecessary `div` wrappers.

❌ BAD: Redundant `div`s around Chakra components.
```javascript
<div>
  <Text>Hello</Text>
</div>
```

✅ GOOD: Use Chakra components directly or `Box` when a wrapper is needed for styling.
```javascript
<Text>Hello</Text>
// Or, if styling is needed:
<Box p="2">
  <Text>Hello</Text>
</Box>
```

### 4.2. Use the `as` Prop for Semantic HTML
Ensure your components render as the correct semantic HTML element.

❌ BAD: Using `Box` for everything, losing semantic meaning.
```javascript
<Box onClick={...}>Clickable text</Box> // Renders as <div>, not a button
```

✅ GOOD: Use `as` to render the appropriate HTML tag.
```javascript
<Box as="button" onClick={...}>Clickable text</Box>
<Box as="section" p="8">Section Content</Box>
```

### 4.3. Leverage Built-in Accessibility
Chakra components are accessible by default. Focus on content and semantic usage.

❌ BAD: Manually adding ARIA attributes to standard Chakra components.
```javascript
<Button aria-label="Submit form">Submit</Button> // Redundant for a standard button
```

✅ GOOD: Trust Chakra's defaults; add ARIA only for custom components or complex interactions.
```javascript
<Button>Submit</Button>
```

## 5. Hooks Best Practices

Utilize Chakra's hooks for dynamic and responsive behavior.

### 5.1. `useBreakpointValue` for Responsive Props
Dynamically adjust values based on breakpoints.

❌ BAD: Manual media queries or `isLargerThan` checks.
```javascript
const MyComponent = () => {
  const { width } = useWindowSize(); // Custom hook for window size
  const padding = width > 768 ? '8' : '4';
  return <Box p={padding}>...</Box>;
};
```

✅ GOOD: Use `useBreakpointValue`.
```javascript
import { Box, useBreakpointValue } from '@chakra-ui/react';
const MyComponent = () => {
  const padding = useBreakpointValue({ base: '4', md: '8' });
  return <Box p={padding}>...</Box>;
};
```

### 5.2. `useColorMode` for Dark/Light Mode
Manage color mode consistently across your application.

❌ BAD: Manually tracking theme state or using global CSS variables.
```javascript
// Some custom context or global state for theme
const { themeMode, toggleThemeMode } = useThemeContext();
const bgColor = themeMode === 'dark' ? 'gray.800' : 'white';
<Box bg={bgColor} />
```

✅ GOOD: Use `useColorMode` and `ColorModeScript`.
```javascript
import { Box, Button, useColorMode, ColorModeScript } from '@chakra-ui/react';
import theme from './theme';

// In your root index.html or equivalent, within <head>
// <ColorModeScript initialColorMode={theme.config.initialColorMode} />

const MyComponent = () => {
  const { colorMode, toggleColorMode } = useColorMode();
  return (
    <Box bg={colorMode === 'dark' ? 'gray.800' : 'white'}>
      <Button onClick={toggleColorMode}>
        Toggle {colorMode === 'light' ? 'Dark' : 'Light'}
      </Button>
    </Box>
  );
};
```