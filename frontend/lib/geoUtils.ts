/**
 * Utility to convert latitude/longitude coordinates to grid positions
 * and manage geographic bounds for traffic visualization.
 */

export interface GeoBounds {
  minLat: number
  maxLat: number
  minLon: number
  maxLon: number
}

export interface GridPosition {
  x: number
  y: number
}

// Default bounds (Mumbai area based on actual data)
const DEFAULT_BOUNDS: GeoBounds = {
  minLat: 19.085,
  maxLat: 19.087,
  minLon: 72.866,
  maxLon: 72.873,
}

/**
 * Convert latitude/longitude to grid cell coordinates
 * @param lat - Latitude (can be integer format like 40742511 or float like 40.742511)
 * @param lon - Longitude (can be integer format like -73949134 or float like -73.949134)
 * @param gridSize - Number of cells in grid (default 25x25)
 * @param bounds - Geographic bounds for the map
 * @returns Grid position {x, y} or null if outside bounds
 */
export function latLonToGrid(
  lat: number,
  lon: number,
  gridSize: number = 25,
  bounds: GeoBounds = DEFAULT_BOUNDS
): GridPosition | null {
  // Convert integer coordinates to float if needed
  const floatLat = lat > 180 ? lat / 1_000_000 : lat
  const floatLon = lon > 180 || lon < -180 ? lon / 1_000_000 : lon

  // Check if coordinates are within bounds
  if (floatLat < bounds.minLat || floatLat > bounds.maxLat || floatLon < bounds.minLon || floatLon > bounds.maxLon) {
    return null
  }

  // Normalize to 0-1 range
  const normalizedLat = (floatLat - bounds.minLat) / (bounds.maxLat - bounds.minLat)
  const normalizedLon = (floatLon - bounds.minLon) / (bounds.maxLon - bounds.minLon)

  // Convert to grid coordinates (flip Y because lat increases north)
  const x = Math.floor(normalizedLon * gridSize)
  const y = Math.floor((1 - normalizedLat) * gridSize) // Flip Y axis

  // Clamp to grid bounds
  const clampedX = Math.max(0, Math.min(gridSize - 1, x))
  const clampedY = Math.max(0, Math.min(gridSize - 1, y))

  return { x: clampedX, y: clampedY }
}

/**
 * Convert grid coordinates back to lat/lon (center of cell)
 */
export function gridToLatLon(
  x: number,
  y: number,
  gridSize: number = 25,
  bounds: GeoBounds = DEFAULT_BOUNDS
): { lat: number; lon: number } {
  // Convert to normalized 0-1 (center of cell)
  const normalizedLon = (x + 0.5) / gridSize
  const normalizedLat = 1 - (y + 0.5) / gridSize // Flip Y axis

  // Scale to actual coordinates
  const lat = bounds.minLat + normalizedLat * (bounds.maxLat - bounds.minLat)
  const lon = bounds.minLon + normalizedLon * (bounds.maxLon - bounds.minLon)

  return { lat, lon }
}

/**
 * Calculate automatic bounds from a set of coordinates
 */
export function calculateBounds(
  coordinates: Array<{ latitude: number; longitude: number }>,
  padding: number = 0.01 // ~1km padding
): GeoBounds {
  if (coordinates.length === 0) {
    return DEFAULT_BOUNDS
  }

  // Convert integer coordinates to float if needed
  const lats = coordinates.map((c) => (c.latitude > 180 ? c.latitude / 1_000_000 : c.latitude))
  const lons = coordinates.map((c) => (c.longitude > 180 || c.longitude < -180 ? c.longitude / 1_000_000 : c.longitude))

  return {
    minLat: Math.min(...lats) - padding,
    maxLat: Math.max(...lats) + padding,
    minLon: Math.min(...lons) - padding,
    maxLon: Math.max(...lons) + padding,
  }
}
