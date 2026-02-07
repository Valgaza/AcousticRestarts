/**
 * Hook to fetch and integrate live traffic predictions into the grid
 */

'use client'

import { useState, useEffect, useCallback } from 'react'
import { fetchCurrentPredictions, type LocationPrediction, type HorizonPrediction } from '@/lib/trafficApi'
import { latLonToGrid, calculateBounds, type GeoBounds } from '@/lib/geoUtils'
import type { GridCell } from './useTrafficSimulation'

const GRID_SIZE = 25
const REFRESH_INTERVAL = 30000 // 30 seconds

export type TimeHorizon = 't+1h' | 't+2h' | 't+3h' | 't+4h' | 't+5h' | 't+6h'

interface LiveTrafficState {
  grid: GridCell[][]
  predictions: LocationPrediction[]
  bounds: GeoBounds
  isLoading: boolean
  error: string | null
  lastUpdate: Date | null
  selectedHorizon: TimeHorizon
  isPlaying: boolean // Auto-cycle through horizons
  speed: number // Playback speed (seconds per horizon)
}

export function useLiveTraffic() {
  const [state, setState] = useState<LiveTrafficState>({
    grid: initializeEmptyGrid(),
    predictions: [],
    bounds: {
      minLat: 40.700,
      maxLat: 40.800,
      minLon: -74.020,
      maxLon: -73.940,
    },
    isLoading: false,
    error: null,
    lastUpdate: null,
    selectedHorizon: 't+1h', // Default to 1 hour ahead
    isPlaying: false,
    speed: 3, // 3 seconds per horizon
  })

  const [autoRefresh, setAutoRefresh] = useState(true)

  // Initialize empty grid
  function initializeEmptyGrid(): GridCell[][] {
    const grid: GridCell[][] = []
    for (let x = 0; x < GRID_SIZE; x++) {
      grid[x] = []
      for (let y = 0; y < GRID_SIZE; y++) {
        grid[x][y] = {
          x,
          y,
          congestionLevel: 0,
          speed: 60,
          predictedCongestion: 0,
        }
      }
    }
    return grid
  }

  // Map predictions to grid for a specific time horizon
  const mapPredictionsToGrid = useCallback(
    (predictions: LocationPrediction[], bounds: GeoBounds, selectedHorizon: TimeHorizon): GridCell[][] => {
      console.log(`📊 Mapping ${predictions.length} predictions for horizon: ${selectedHorizon}`)
      const newGrid = initializeEmptyGrid()
      
      // Create a map to accumulate congestion values per cell
      const cellData = new Map<string, { sum: number; count: number }>()
      let horizonDataFound = 0

      predictions.forEach((pred) => {
        const gridPos = latLonToGrid(pred.latitude, pred.longitude, GRID_SIZE, bounds)
        if (!gridPos) return

        // Find the selected time horizon
        const horizonData = pred.horizons.find((h) => h.horizon === selectedHorizon)
        if (!horizonData) {
          console.warn(`⚠️ No horizon data found for ${selectedHorizon} at location`, pred.latitude, pred.longitude)
          return
        }
        horizonDataFound++

        const key = `${gridPos.x},${gridPos.y}`
        const existing = cellData.get(key) || { sum: 0, count: 0 }
        cellData.set(key, {
          sum: existing.sum + horizonData.predicted_congestion,
          count: existing.count + 1,
        })
      })

      // Update grid cells with averaged congestion
      cellData.forEach((data, key) => {
        const [x, y] = key.split(',').map(Number)
        const avgCongestion = data.sum / data.count
        
        // Normalize congestion to 0-1 range (from ~-1 to +1)
        const normalizedCongestion = Math.max(0, Math.min(1, (avgCongestion + 1) / 2))
        
        newGrid[x][y] = {
          x,
          y,
          congestionLevel: normalizedCongestion,
          speed: 60 * (1 - normalizedCongestion * 0.7), // Speed decreases with congestion
          predictedCongestion: normalizedCongestion,
        }
      })

      console.log(`✅ Mapped to ${cellData.size} grid cells (${horizonDataFound} predictions with horizon data)`)
      return newGrid
    },
    []
  )

  // Fetch predictions from API
  const fetchPredictions = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }))

    try {
      const response = await fetchCurrentPredictions(100) // Fetch 100 locations
      const predictions = response.predictions

      // Calculate bounds from actual data
      const bounds = calculateBounds(
        predictions.map((p) => ({ latitude: p.latitude, longitude: p.longitude }))
      )

      // Map to grid using current selected horizon
      setState((prev) => {
        const grid = mapPredictionsToGrid(predictions, bounds, prev.selectedHorizon)
        return {
          ...prev,
          grid,
          predictions,
          bounds,
          isLoading: false,
          lastUpdate: new Date(),
        }
      })
    } catch (error) {
      console.error('Failed to fetch predictions:', error)
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }))
    }
  }, [mapPredictionsToGrid])

  // Initial fetch
  useEffect(() => {
    fetchPredictions()
  }, [fetchPredictions])

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      fetchPredictions()
    }, REFRESH_INTERVAL)

    return () => clearInterval(interval)
  }, [autoRefresh, fetchPredictions])

  // Auto-play through horizons
  useEffect(() => {
    if (!state.isPlaying) {
      console.log('Auto-play stopped')
      return
    }

    console.log('Auto-play started, speed:', state.speed, 'seconds')
    const horizons: TimeHorizon[] = ['t+1h', 't+2h', 't+3h', 't+4h', 't+5h', 't+6h']
    
    const interval = setInterval(() => {
      setState((prev) => {
        // Calculate next horizon based on current state
        const currentIndex = horizons.indexOf(prev.selectedHorizon)
        const nextIndex = (currentIndex + 1) % horizons.length
        const nextHorizon = horizons[nextIndex]
        
        console.log('Cycling:', prev.selectedHorizon, '→', nextHorizon)
        
        if (prev.predictions.length === 0) {
          return { ...prev, selectedHorizon: nextHorizon }
        }
        
        const grid = mapPredictionsToGrid(prev.predictions, prev.bounds, nextHorizon)
        return { ...prev, selectedHorizon: nextHorizon, grid }
      })
    }, state.speed * 1000)

    return () => {
      console.log('Clearing auto-play interval')
      clearInterval(interval)
    }
  }, [state.isPlaying, state.speed, mapPredictionsToGrid])

  const toggleAutoRefresh = useCallback(() => {
    setAutoRefresh((prev) => !prev)
  }, [])

  const manualRefresh = useCallback(() => {
    fetchPredictions()
  }, [fetchPredictions])

  // Change selected time horizon and re-map grid
  const setSelectedHorizon = useCallback((horizon: TimeHorizon) => {
    setState((prev) => {
      if (prev.predictions.length === 0) {
        return { ...prev, selectedHorizon: horizon }
      }
      const grid = mapPredictionsToGrid(prev.predictions, prev.bounds, horizon)
      return { ...prev, selectedHorizon: horizon, grid }
    })
  }, [mapPredictionsToGrid])

  // Toggle play/pause for horizon cycling
  const togglePlayPause = useCallback(() => {
    setState((prev) => {
      console.log('Live traffic play toggled:', !prev.isPlaying)
      return { ...prev, isPlaying: !prev.isPlaying }
    })
  }, [])

  // Set playback speed (seconds per horizon)
  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({ ...prev, speed }))
  }, [])

  return {
    ...state,
    autoRefresh,
    toggleAutoRefresh,
    manualRefresh,
    setSelectedHorizon,
    togglePlayPause,
    setSpeed,
  }
}
