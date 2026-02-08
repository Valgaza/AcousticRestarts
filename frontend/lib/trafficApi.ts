/**
 * API service for fetching traffic predictions from the backend
 */

export interface HorizonPrediction {
  horizon: string // "t+1h", "t+2h", etc.
  DateTime: string
  predicted_congestion: number // Normalized value (~-1 to +1)
}

export interface LocationPrediction {
  latitude: number // Integer format (40742511 = 40.742511°)
  longitude: number // Integer format (-73949134 = -73.949134°)
  horizons: HorizonPrediction[] // 6 time horizons
}

export interface PredictionResponse {
  predictions: LocationPrediction[]
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/**
 * Fetch current traffic predictions (multi-horizon)
 * @param locations - Number of locations to fetch
 * @returns Predictions with 6 time horizons per location
 */
export async function fetchCurrentPredictions(locations: number = 50): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/predictions/current?locations=${locations}`)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

/**
 * Fetch batch predictions
 * @param locations - Number of locations to fetch
 * @param baseHour - Base hour for predictions (0-23)
 * @returns Predictions with 6 time horizons per location
 */
export async function fetchBatchPredictions(
  locations: number = 50,
  baseHour?: number
): Promise<PredictionResponse> {
  const params = new URLSearchParams({ locations: locations.toString() })
  if (baseHour !== undefined) {
    params.append('base_hour', baseHour.toString())
  }

  const response = await fetch(`${API_BASE_URL}/predictions/batch?${params}`)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}

/**
 * Health check
 */
export async function checkApiHealth(): Promise<{ status: string; timestamp: string }> {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error('API health check failed')
  }
  return response.json()
}

/**
 * Reload forecasts from JSON file
 * Call this after your ML model updates forecasts.json
 */
export async function reloadForecasts(): Promise<{ status: string; message: string; locations: number }> {
  const response = await fetch(`${API_BASE_URL}/reload`, { method: 'POST' })
  if (!response.ok) {
    throw new Error('Failed to reload forecasts')
  }
  return response.json()
}

/**
 * Upload a CSV file for ML model processing
 * @param file - CSV file to upload
 * @returns Upload status and file info
 */
export async function uploadCsvFile(file: File): Promise<{
  status: string
  message: string
  filename: string
  size_bytes: number
}> {
  // Validate file type on client side
  if (!file.name.endsWith('.csv')) {
    throw new Error('Only CSV files are allowed. Please select a .csv file.')
  }

  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload-csv`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to upload CSV file')
  }

  return response.json()
}

/**
 * Fetch all grid frames with congestion data
 */
export interface GridCell {
  row: number
  col: number
  predicted_congestion_level: number
}

export interface GridFrame {
  DateTime: string
  cells: GridCell[]
}

export interface GridFramesResponse {
  metadata: {
    grid_size: number
    bounds: {
      lat_min: number
      lat_max: number
      lon_min: number
      lon_max: number
    }
    total_timestamps: number
    cells_per_timestamp: number
  }
  frames: GridFrame[]
  total_frames: number
}

export async function fetchGridFrames(): Promise<GridFramesResponse> {
  const response = await fetch(`${API_BASE_URL}/grid-frames`)
  if (!response.ok) {
    throw new Error('Failed to fetch grid frames')
  }
  return response.json()
}

/**
 * Fetch a specific grid frame by index
 */
export async function fetchGridFrame(frameIndex: number): Promise<{
  metadata: any
  frame: GridFrame
  frame_index: number
  total_frames: number
}> {
  const response = await fetch(`${API_BASE_URL}/grid-frame/${frameIndex}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch frame ${frameIndex}`)
  }
  return response.json()
}

/**
 * Route optimization request/response types
 */
export interface RouteRequest {
  start_node: number
  end_node: number
  timestamp: string // ISO format: "2026-02-08T03:00:00"
}

export interface RouteAssignment {
  request_idx: number
  chosen_route: number[] // Edge IDs
  route_nodes: number[] // Node IDs
  total_cost: number
  alternatives_considered: number
}

export interface RouteOptimizationResponse {
  assignments: RouteAssignment[]
  output_csv_path: string
}

/**
 * Submit route optimization requests
 * @param requests - Array of route optimization requests
 * @returns Route assignments and output CSV path
 */
export async function optimizeRoutes(requests: RouteRequest[]): Promise<RouteOptimizationResponse> {
  const response = await fetch(`${API_BASE_URL}/optimize-routes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ requests }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Route optimization failed')
  }

  return response.json()
}
