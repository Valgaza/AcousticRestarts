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
