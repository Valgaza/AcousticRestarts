/**
 * Hook to fetch and display grid-based traffic predictions
 */

'use client'

import { useState, useEffect, useCallback } from 'react'
import { fetchGridFrames, type GridFrame, type GridFramesResponse } from '@/lib/trafficApi'
import type { GridCell as TrafficGridCell } from './useTrafficSimulation'

const GRID_SIZE = 25
const REFRESH_INTERVAL = 30000 // 30 seconds
const FRAME_CYCLE_INTERVAL = 3000 // 3 seconds per frame

interface GridTrafficState {
  grid: TrafficGridCell[][]
  frames: GridFrame[]
  currentFrameIndex: number
  metadata: any
  isLoading: boolean
  error: string | null
  lastUpdate: Date | null
  isPlaying: boolean
}

export function useGridTraffic() {
  const [state, setState] = useState<GridTrafficState>({
    grid: initializeEmptyGrid(),
    frames: [],
    currentFrameIndex: 0,
    metadata: null,
    isLoading: false,
    error: null,
    lastUpdate: null,
    isPlaying: false,
  })

  const [autoRefresh, setAutoRefresh] = useState(true)

  // Initialize empty grid
  function initializeEmptyGrid(): TrafficGridCell[][] {
    const grid: TrafficGridCell[][] = []
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

  // Convert frame cells to grid format
  const frameToGrid = useCallback((frame: GridFrame): TrafficGridCell[][] => {
    const newGrid = initializeEmptyGrid()
    
    frame.cells.forEach((cell) => {
      const { row, col, predicted_congestion_level } = cell
      
      // Ensure row and col are within bounds
      if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE) {
        // Grid uses [x][y] where x is column, y is row
        newGrid[col][row] = {
          x: col,
          y: row,
          congestionLevel: predicted_congestion_level,
          speed: 60 * (1 - predicted_congestion_level * 0.7),
          predictedCongestion: predicted_congestion_level,
        }
      }
    })
    
    return newGrid
  }, [])

  // Fetch grid frames from API
  const fetchFrames = useCallback(async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }))

    try {
      const response = await fetchGridFrames()
      
      if (response.frames.length === 0) {
        throw new Error('No grid frames available')
      }

      // Set initial grid to first frame
      const initialGrid = frameToGrid(response.frames[0])

      setState((prev) => ({
        ...prev,
        frames: response.frames,
        metadata: response.metadata,
        grid: initialGrid,
        currentFrameIndex: 0,
        isLoading: false,
        lastUpdate: new Date(),
      }))
    } catch (error) {
      console.error('Failed to fetch grid frames:', error)
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }))
    }
  }, [frameToGrid])

  // Initial fetch
  useEffect(() => {
    fetchFrames()
  }, [fetchFrames])

  // Auto-refresh frames from API
  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(() => {
      fetchFrames()
    }, REFRESH_INTERVAL)

    return () => clearInterval(interval)
  }, [autoRefresh, fetchFrames])

  // Auto-cycle through frames
  useEffect(() => {
    if (!state.isPlaying || state.frames.length === 0) return

    const interval = setInterval(() => {
      setState((prev) => {
        const nextIndex = (prev.currentFrameIndex + 1) % prev.frames.length
        const nextGrid = frameToGrid(prev.frames[nextIndex])
        
        return {
          ...prev,
          currentFrameIndex: nextIndex,
          grid: nextGrid,
        }
      })
    }, FRAME_CYCLE_INTERVAL)

    return () => clearInterval(interval)
  }, [state.isPlaying, state.frames.length, frameToGrid])

  const toggleAutoRefresh = useCallback(() => {
    setAutoRefresh((prev) => !prev)
  }, [])

  const manualRefresh = useCallback(() => {
    fetchFrames()
  }, [fetchFrames])

  const setFrameIndex = useCallback((index: number) => {
    setState((prev) => {
      if (index < 0 || index >= prev.frames.length) return prev
      
      const newGrid = frameToGrid(prev.frames[index])
      return {
        ...prev,
        currentFrameIndex: index,
        grid: newGrid,
      }
    })
  }, [frameToGrid])

  const togglePlayPause = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: !prev.isPlaying }))
  }, [])

  return {
    ...state,
    autoRefresh,
    toggleAutoRefresh,
    manualRefresh,
    setFrameIndex,
    togglePlayPause,
  }
}
