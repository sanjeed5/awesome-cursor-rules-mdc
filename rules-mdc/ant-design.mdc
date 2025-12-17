---
description: Definitive guidelines for building consistent, performant, and maintainable UIs with Ant Design v6.x, focusing on design tokens, layout, and component usage.
globs: **/*
---
# ant-design Best Practices

This guide outlines the definitive best practices for using Ant Design v6.x in our projects. Adhering to these standards ensures consistency, accessibility, and maintainability across our codebase.

## 1. Theming: Design Token Driven

Always customize Ant Design's appearance using its design token system via `ConfigProvider`. Avoid direct CSS overrides for visual properties that can be controlled by tokens.

### ✅ GOOD: Centralized Token Customization

Create a `theme.ts` file to export your theme configuration and apply it at the root of your application. Prefer `seedToken` adjustments for global changes and `componentToken` for specific component overrides. Utilize preset algorithms for light/dark mode.

```typescript
// src/theme.ts
import { theme } from 'antd';
import type { ThemeConfig } from 'antd';

export const appTheme: ThemeConfig = {
  algorithm: theme.defaultAlgorithm, // Or theme.darkAlgorithm, [theme.defaultAlgorithm, theme.darkAlgorithm]
  token: {
    colorPrimary: '#0050B3', // Our primary brand color
    colorSuccess: '#52C41A',
    fontSize: 14,
    borderRadius: 4,
    // Disable motion globally for accessibility/performance
    motion: false,
  },
  components: {
    Button: {
      colorPrimary: '#1890FF', // Override primary button color
      algorithm: true, // Enable algorithm for component token derivation
    },
    Layout: {
      headerBg: '#001529',
      footerBg: '#F0F2F5',
    },
  },
};
```

```tsx
// src/App.tsx (or root layout)
import React from 'react';
import { ConfigProvider, App as AntdApp } from 'antd';
import { appTheme } from './theme';
import HomePage from './pages/HomePage';

const App: React.FC = () => {
  return (
    <ConfigProvider theme={appTheme}>
      {/* Use AntdApp for context-aware static methods (message, Modal, notification) */}
      <AntdApp>
        <HomePage />
      </AntdApp>
    </ConfigProvider>
  );
};

export default App;
```

### ❌ BAD: Direct CSS Overrides

Avoid overriding Ant Design component styles with custom CSS classes or inline styles for properties that are themeable. This leads to inconsistent UIs and breaks theme updates.

```css
/* ❌ BAD: Avoid this for themeable properties */
.ant-btn-primary {
  background-color: #f00 !important; /* Breaks theme */
  border-color: #f00 !important;
}
```

## 2. Layout: Unified Grid System

Adhere to Ant Design's 8-pixel grid unit and the unified 1440px canvas width. Use `Row` and `Col` components for structural layout, leveraging their `span` and `gutter` props.

### ✅ GOOD: Responsive Layout with Grid

```tsx
import React from 'react';
import { Layout, Row, Col } from 'antd';

const { Header, Content, Footer } = Layout;

const MyLayout: React.FC = () => (
  <Layout style={{ minHeight: '100vh' }}>
    <Header style={{ backgroundColor: '#001529' }}>
      {/* Header content */}
    </Header>
    <Content style={{ padding: '0 50px', maxWidth: 1440, margin: '16px auto' }}>
      <Row gutter={[16, 16]}> {/* 16px horizontal and vertical gutter */}
        <Col span={24}>
          <h1>Welcome to our Application</h1>
        </Col>
        <Col xs={24} sm={12} md={8}> {/* Responsive columns */}
          <div style={{ background: '#fff', padding: 24, minHeight: 120 }}>
            Card 1
          </div>
        </Col>
        <Col xs={24} sm={12} md={8}>
          <div style={{ background: '#fff', padding: 24, minHeight: 120 }}>
            Card 2
          </div>
        </Col>
        <Col xs={24} sm={24} md={8}>
          <div style={{ background: '#fff', padding: 24, minHeight: 120 }}>
            Card 3
          </div>
        </Col>
      </Row>
    </Content>
    <Footer style={{ textAlign: 'center' }}>
      Ant Design ©{new Date().getFullYear()} Created by Ant UED
    </Footer>
  </Layout>
);

export default MyLayout;
```

### ❌ BAD: Hardcoding Layout Dimensions

Avoid using fixed pixel values or custom CSS for spacing and alignment that should be handled by the grid system or design tokens.

```tsx
// ❌ BAD: Avoid custom margins for layout
<div style={{ marginLeft: '20px', paddingRight: '30px' }}>
  {/* Content */}
</div>
```

## 3. Component Usage: Standardized & Accessible

Import components directly from `antd`. Always use TypeScript and configure ESLint with `eslint-plugin-antd` to enforce best practices. Prioritize accessibility by providing necessary `aria-*` attributes.

### ✅ GOOD: Proper Imports and Accessibility

```tsx
import React from 'react';
import { Button, Space, Typography } from 'antd';
import { PlusOutlined } from '@ant-design/icons';

const { Text } = Typography;

const MyComponent: React.FC = () => {
  const handleClick = () => {
    console.log('Button clicked');
  };

  return (
    <Space>
      <Button type="primary" onClick={handleClick} icon={<PlusOutlined />} aria-label="Add new item">
        Add Item
      </Button>
      <Text strong>Important information.</Text>
    </Space>
  );
};

export default MyComponent;
```

### ❌ BAD: Missing Accessibility Attributes

Interactive components must have appropriate `aria-*` attributes for screen reader users.

```tsx
// ❌ BAD: Missing aria-label for icon-only button
<Button icon={<PlusOutlined />} onClick={() => {}} />
```

## 4. Internationalization & SSR

Wrap your root component with `ConfigProvider` to set the locale and theme. For Server-Side Rendering (SSR), follow the official Ant Design SSR guide to prevent hydration mismatches.

### ✅ GOOD: Global Locale Configuration

```tsx
// src/main.tsx (or root entry)
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider } from 'antd';
import enUS from 'antd/locale/en_US'; // Import desired locale
import App from './App';
import { appTheme } from './theme';

const container = document.getElementById('root');
const root = createRoot(container!);

root.render(
  <React.StrictMode>
    <ConfigProvider locale={enUS} theme={appTheme}>
      <App />
    </ConfigProvider>
  </React.StrictMode>
);
```

## 5. Performance: Zero Runtime & Tree Shaking

Leverage Ant Design's `zeroRuntime` mode (v6.0.0+) in production for improved performance by pre-generating styles. Ensure your build setup supports tree shaking for `antd` to minimize bundle size.

### ✅ GOOD: Zero Runtime Mode in Production

For production builds, enable `zeroRuntime` and import the pre-generated CSS.

```tsx
// src/App.tsx
import React from 'react';
import { ConfigProvider, App as AntdApp } from 'antd';
import { appTheme } from './theme';
import HomePage from './pages/HomePage';

// In production, import the static CSS generated by @ant-design/static-style-extract
// import 'antd/dist/antd.css'; // Or your custom extracted CSS file

const App: React.FC = () => {
  // In a real app, you'd conditionally apply zeroRuntime based on NODE_ENV
  const isProduction = process.env.NODE_ENV === 'production';

  return (
    <ConfigProvider theme={{ ...appTheme, zeroRuntime: isProduction }}>
      <AntdApp>
        <HomePage />
      </AntdApp>
    </ConfigProvider>
  );
};

export default App;
```

### ❌ BAD: Importing Full CSS with Runtime Styling

If `zeroRuntime` is `false` (default for development), avoid importing `antd/dist/antd.css` as it will lead to duplicate styles and larger bundle sizes.

```tsx
// ❌ BAD: If zeroRuntime is false, this is redundant and harmful
import 'antd/dist/antd.css';
```

## 6. Hooks Best Practices

Utilize Ant Design's hooks for accessing theme tokens and managing context-aware static methods.

### ✅ GOOD: Using `useToken` and `App` Component

```tsx
import React from 'react';
import { Button, theme, message } from 'antd';
import { App as AntdApp } from 'antd'; // Import App component for static methods

const MyThemedComponent: React.FC = () => {
  const { token } = theme.useToken(); // Access current theme tokens
  const [messageApi, contextHolder] = message.useMessage(); // For context-aware message

  const showInfo = () => {
    messageApi.info(`Primary color: ${token.colorPrimary}`);
  };

  return (
    <div>
      {contextHolder} {/* Important: Render contextHolder */}
      <Button onClick={showInfo} style={{ backgroundColor: token.colorPrimary, color: token.colorTextLightSolid }}>
        Show Primary Color
      </Button>
    </div>
  );
};

// Wrap components that use static methods or useToken inside <AntdApp>
const RootComponent: React.FC = () => (
  <AntdApp>
    <MyThemedComponent />
  </AntdApp>
);

export default RootComponent;
```

### ❌ BAD: Static Methods Without Context

Calling `message.info()` or `Modal.confirm()` directly without `App` component or `useMessage`/`useModal` hooks will result in them not inheriting `ConfigProvider` context (e.g., theme, locale).

```tsx
// ❌ BAD: This message will not inherit ConfigProvider context
import { message, Button } from 'antd';

const BadComponent: React.FC = () => (
  <Button onClick={() => message.info('Hello')}>
    Show Message
  </Button>
);
```