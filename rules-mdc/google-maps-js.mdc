---
description: This guide provides definitive best practices for integrating and managing the Google Maps JavaScript API in modern web applications, focusing on security, performance, and maintainability.
globs: **/*.{js,jsx}
---
# google-maps-js Best Practices

This document outlines the definitive guidelines for working with the Google Maps JavaScript API. Adhering to these practices ensures secure, performant, and maintainable map-centric applications.

## 1. Code Organization and Structure

Encapsulate map logic to keep components clean and reusable. Avoid direct global `google.maps` object manipulation within React components.

### ✅ GOOD: Custom Hooks for Map Logic (React)

Abstract map initialization and interactions into custom hooks.

```jsx
// hooks/useGoogleMap.js
import { useEffect, useRef, useState } from 'react';
import { Loader } from '@googlemaps/js-api-loader';

const API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY; // Securely load API key

export function useGoogleMap(mapContainerRef, options) {
  const [map, setMap] = useState(null);
  const loaderRef = useRef(null);

  useEffect(() => {
    if (!mapContainerRef.current || !API_KEY) return;

    if (!loaderRef.current) {
      loaderRef.current = new Loader({
        apiKey: API_KEY,
        version: 'weekly', // Use 'weekly' for latest features, 'quarterly' for stability
        // Do NOT specify 'libraries' here unless absolutely necessary for caching.
        // Prefer dynamic importLibrary() calls.
      });
    }

    const initMap = async () => {
      try {
        const { Map } = await loaderRef.current.importLibrary('maps');
        const newMap = new Map(mapContainerRef.current, {
          center: { lat: -34.397, lng: 150.644 },
          zoom: 8,
          mapId: 'YOUR_MAP_ID', // Always use a Map ID
          ...options,
        });
        setMap(newMap);
      } catch (error) {
        console.error('Error loading Google Maps:', error);
        // Implement user-facing error message, e.g., show a fallback UI
      }
    };

    initMap();

    // Cleanup: Google Maps API doesn't have a direct 'destroy' method for Map instances
    // but unmounting the container and nulling the map reference helps.
    return () => {
      setMap(null); // Clear map instance on unmount
    };
  }, [mapContainerRef, options]);

  return map;
}

// components/MyMapComponent.jsx
import React, { useRef, useEffect } from 'react';
import { useGoogleMap } from '../hooks/useGoogleMap';

export function MyMapComponent() {
  const mapRef = useRef(null);
  const map = useGoogleMap(mapRef, { zoom: 10, center: { lat: 34.0522, lng: -118.2437 } });

  useEffect(() => {
    if (map) {
      // Map is loaded, add markers, listeners, etc.
      console.log('Map loaded:', map);
      const marker = new google.maps.marker.AdvancedMarkerElement({
        map,
        position: map.getCenter(),
        title: 'Hello World',
      });
      // Remember to clean up markers/overlays if they are not managed by the map itself
    }
  }, [map]);

  return <div ref={mapRef} style={{ height: '500px', width: '100%' }} />;
}
```

## 2. Security

API key security is paramount. Never expose your API keys directly in client-side code.

### ❌ BAD: Hardcoding API Keys

```javascript
// In a client-side JS file
const API_KEY = "YOUR_HARDCODED_API_KEY_HERE"; // EXPOSED!
const loader = new Loader({ apiKey: API_KEY });
```

### ✅ GOOD: Secure API Key Management

Store API keys as environment variables and restrict them in the Google Cloud Console.

```javascript
// .env.local (for Next.js/Vite, etc.)
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=AIzaSy...

// In your application code (e.g., useGoogleMap.js)
const API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

// Google Cloud Console: Always restrict API keys by HTTP referrer (your domain)
// or by application (Android/iOS package names).
// Use separate keys for different projects/environments to isolate failures.
```

## 3. Loading the API

Always use modern, performant methods for loading the Maps JavaScript API.

### ❌ BAD: Synchronous Script Tag

```html
<!-- Loads all libraries upfront, blocks rendering -->
<script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&callback=initMap"></script>
```

### ✅ GOOD: Dynamic Library Import with `js-api-loader`

This allows lazy loading of specific libraries as needed, reducing initial bundle size and improving page load performance.

```javascript
import { Loader } from '@googlemaps/js-api-loader';

const loader = new Loader({
  apiKey: process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY,
  version: 'weekly',
});

async function initializeMapAndPlaces() {
  try {
    const { Map } = await loader.importLibrary('maps');
    const { AdvancedMarkerElement } = await loader.importLibrary('marker');
    const { AutocompleteService } = await loader.importLibrary('places');

    const map = new Map(document.getElementById('map'), { /* ... */ });
    const marker = new AdvancedMarkerElement({ map, /* ... */ });
    const autocompleteService = new AutocompleteService();
    // ... use services
  } catch (error) {
    console.error('Failed to load Google Maps libraries:', error);
  }
}

initializeMapAndPlaces();
```

## 4. Event Handling

Manage event listeners carefully to prevent memory leaks and ensure proper cleanup.

### ❌ BAD: Unmanaged Event Listeners

```javascript
// In a React component, without cleanup
useEffect(() => {
  if (map) {
    map.addListener('click', (e) => {
      console.log('Map clicked at:', e.latLng.toString());
    });
    // This listener will persist even if the component unmounts
  }
}, [map]);
```

### ✅ GOOD: Managed Event Listeners with Cleanup

Use `google.maps.event.addListener` and store the listener handle for later removal with `google.maps.event.removeListener`. In React, this is handled in `useEffect`'s cleanup function.

```javascript
useEffect(() => {
  if (map) {
    const clickListener = map.addListener('click', (e) => {
      console.log('Map clicked at:', e.latLng.toString());
    });

    const dragEndListener = map.addListener('dragend', () => {
      console.log('Map dragged to:', map.getCenter().toString());
    });

    // Cleanup function: remove listeners when component unmounts or dependencies change
    return () => {
      google.maps.event.removeListener(clickListener);
      google.maps.event.removeListener(dragEndListener);
    };
  }
}, [map]);
```

## 5. Performance Considerations

Optimize map rendering and API requests to ensure a smooth user experience.

### 5.1 Marker Management

For applications with many markers, implement clustering or only render visible markers.

### ✅ GOOD: Marker Clustering

Use a helper library like `@googlemaps/markerclusterer` for efficient marker management.

```javascript
import { MarkerClusterer } from '@googlemaps/markerclusterer';

async function addClusteredMarkers(map, locations) {
  const { AdvancedMarkerElement } = await loader.importLibrary('marker');
  const markers = locations.map(loc => new AdvancedMarkerElement({ position: loc }));

  new MarkerClusterer({ map, markers });
}
```

### 5.2 Debouncing/Throttling Map Events

Avoid excessive re-renders or API calls on continuous events like `mousemove` or `drag`.

### ✅ GOOD: Debouncing `idle` Event

```javascript
import { debounce } from 'lodash'; // Or implement your own debounce utility

useEffect(() => {
  if (map) {
    const handleMapIdle = debounce(() => {
      const center = map.getCenter();
      const zoom = map.getZoom();
      console.log(`Map idle. Center: ${center.toString()}, Zoom: ${zoom}`);
      // Trigger API calls or state updates here (e.g., fetch new data for current viewport)
    }, 500); // Wait 500ms after map stops moving

    const idleListener = map.addListener('idle', handleMapIdle);

    return () => {
      google.maps.event.removeListener(idleListener);
      handleMapIdle.cancel(); // Cancel any pending debounced calls
    };
  }
}, [map]);
```

## 6. Request/Response Patterns & Rate Limiting

All web service requests must be asynchronous, use HTTPS, and implement robust error handling with exponential backoff.

### ❌ BAD: Synchronous Requests & No Backoff

```javascript
// This is a conceptual BAD example, as browser JS typically prevents synchronous XHR to external domains.
// The core issue is making repeated, unmanaged requests to web services.
function getGeolocation(location) {
  // ... make direct fetch call without retry logic ...
  // This will fail on transient errors and could hit rate limits quickly.
}
```

### ✅ GOOD: Asynchronous Requests with Exponential Backoff

For server-side or client-side web service calls (e.g., Geocoding, Places API *web services*), implement exponential backoff.

```javascript
async function fetchWithExponentialBackoff(url, maxRetries = 5, initialDelay = 100) {
  let delay = initialDelay;
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        // Handle 4XX/5XX errors
        if (response.status >= 400 && response.status < 500 && response.status !== 429) {
          // Client errors (e.g., bad request, unauthorized) should not be retried
          throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        // Server errors or rate limits (429) should be retried
        throw new Error(`Transient API Error: ${response.status} ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.warn(`Attempt ${i + 1} failed: ${error.message}. Retrying in ${delay / 1000}s...`);
      if (i === maxRetries - 1) throw error; // Re-throw after last retry
      await new Promise(resolve => setTimeout(resolve, delay));
      delay *= 2; // Exponential increase
    }
  }
}

// Example usage for a Geocoding API call (server-side recommended for API keys)
async function geocodeAddress(address) {
  const encodedAddress = encodeURIComponent(address);
  const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=${process.env.GOOGLE_MAPS_SERVER_API_KEY}`;
  try {
    const data = await fetchWithExponentialBackoff(url);
    if (data.status === 'OK') {
      return data.results[0].geometry.location;
    } else {
      throw new Error(`Geocoding failed: ${data.status} - ${data.error_message || 'Unknown error'}`);
    }
  } catch (error) {
    console.error('Final Geocoding error:', error);
    // Notify user or log for investigation
  }
}
```

### 6.1 URL Encoding

Always URL-encode parameters for web service requests, especially before signing if applicable.

```javascript
// ❌ BAD: Unencoded parameters can break requests or lead to incorrect results
const address = "5th&Main St.";
const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${address}&key=...`;

// ✅ GOOD: Properly encoded parameters
const address = "5th&Main St.";
const encodedAddress = encodeURIComponent(address); // "5th%26Main%20St."
const url = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=...`;
```

## 7. Common Pitfalls and Gotchas

### 7.1 Map ID Usage

Always use a Map ID when initializing a map. This enables cloud-based map styling and future features.

```javascript
// ❌