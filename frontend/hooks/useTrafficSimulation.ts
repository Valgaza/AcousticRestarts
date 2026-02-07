'use client'

import { useState, useCallback, useEffect } from 'react'

export interface GridCell {
  x: number
  y: number
  congestionLevel: number
  speed: number
  predictedCongestion: number
}

export interface RouteMetric {
  id: string
  name: string
  currentSpeed: number
  freeFlowSpeed: number
  sparklineData: number[]
}

interface TrafficSimulationState {
  grid: GridCell[][]
  routes: RouteMetric[]
  isPlaying: boolean
  speed: number
  weather: 'clear' | 'rain' | 'snow' | 'fog'
  currentTime: number
  avgCitySpeed: number
  activeBottlenecks: number
  weatherPenalty: number
}

const GRID_SIZE = 25
const SIMULATION_TIME_STEP = 1 // hours
const FREE_FLOW_SPEED = 60 // km/h

export function useTrafficSimulation() {
  const [state, setState] = useState<TrafficSimulationState>({
    grid: initializeGrid(),
    routes: initializeRoutes(),
    isPlaying: false,
    speed: 1,
    weather: 'clear',
    currentTime: 8, // 8:00 AM
    avgCitySpeed: 45,
    activeBottlenecks: 3,
    weatherPenalty: 0,
  })

  // Rush hour pattern using sine wave (peaks at 8:00 and 18:00)
  const getRushHourFactor = useCallback((hour: number) => {
    const hour24 = hour % 24
    const morningPeak = Math.sin(((hour24 - 6) * Math.PI) / 6) * 0.5 + 0.5 // 6:00-12:00
    const eveningPeak = Math.sin(((hour24 - 16) * Math.PI) / 6) * 0.5 + 0.5 // 16:00-22:00
    return Math.max(morningPeak, eveningPeak, 0.2) // Minimum 0.2 congestion
  }, [])

  // Weather impact on speeds
  const getWeatherPenalty = useCallback((weather: string) => {
    switch (weather) {
      case 'rain':
        return 0.2
      case 'snow':
        return 0.35
      case 'fog':
        return 0.15
      default:
        return 0
    }
  }, [])

  // Initialize grid with random congestion
  function initializeGrid(): GridCell[][] {
    const grid: GridCell[][] = []
    for (let x = 0; x < GRID_SIZE; x++) {
      grid[x] = []
      for (let y = 0; y < GRID_SIZE; y++) {
        grid[x][y] = {
          x,
          y,
          congestionLevel: Math.random() * 0.3,
          speed: FREE_FLOW_SPEED,
          predictedCongestion: Math.random() * 0.3,
        }
      }
    }
    return grid
  }

  // Initialize route metrics
  function initializeRoutes(): RouteMetric[] {
    return [
      {
        id: 'i90-east',
        name: 'I-90 Eastbound',
        currentSpeed: 50,
        freeFlowSpeed: 65,
        sparklineData: Array.from({ length: 20 }, () => Math.random() * 60 + 40),
      },
      {
        id: 'downtown-loop',
        name: 'Downtown Loop',
        currentSpeed: 35,
        freeFlowSpeed: 50,
        sparklineData: Array.from({ length: 20 }, () => Math.random() * 50 + 30),
      },
      {
        id: 'hwy405',
        name: 'Highway 405',
        currentSpeed: 45,
        freeFlowSpeed: 70,
        sparklineData: Array.from({ length: 20 }, () => Math.random() * 70 + 35),
      },
    ]
  }

  // Update simulation based on time and weather
  const updateSimulation = useCallback((newTime?: number) => {
    setState((prevState) => {
      const time = newTime !== undefined ? newTime : prevState.currentTime
      const rushHour = getRushHourFactor(time)
      const weatherPenalty = getWeatherPenalty(prevState.weather)
      const totalPenalty = rushHour + weatherPenalty

      // Update grid cells based on rush hour and weather
      const newGrid = prevState.grid.map((row) =>
        row.map((cell) => {
          // Add some randomness but follow the rush hour pattern
          const baseLevel = rushHour + Math.random() * 0.3 - 0.15
          const congestionLevel = Math.max(0, Math.min(1, baseLevel + weatherPenalty))
          const speedFactor = 1 - congestionLevel * 0.7
          const speed = FREE_FLOW_SPEED * speedFactor
          const predictedLevel = Math.max(0, Math.min(1, baseLevel + 0.1))

          return {
            ...cell,
            congestionLevel,
            speed,
            predictedCongestion: predictedLevel,
          }
        })
      )

      // Calculate metrics
      const avgSpeed = Math.round(
        newGrid.flat().reduce((sum, cell) => sum + cell.speed, 0) / (GRID_SIZE * GRID_SIZE)
      )

      const bottleneckCount = newGrid
        .flat()
        .filter((cell) => cell.congestionLevel > 0.65).length

      return {
        ...prevState,
        grid: newGrid,
        currentTime: time,
        avgCitySpeed: avgSpeed,
        activeBottlenecks: Math.ceil(bottleneckCount / 5),
        weatherPenalty: Math.round(weatherPenalty * 100),
      }
    })
  }, [getRushHourFactor, getWeatherPenalty])

  // Animation loop
  useEffect(() => {
    if (!state.isPlaying) return

    const interval = setInterval(() => {
      setState((prevState) => {
        const newTime = (prevState.currentTime + (SIMULATION_TIME_STEP * prevState.speed) / 10) % 24
        return { ...prevState, currentTime: newTime }
      })
      updateSimulation()
    }, 200)

    return () => clearInterval(interval)
  }, [state.isPlaying, updateSimulation])

  const togglePlayPause = useCallback(() => {
    setState((prev) => ({ ...prev, isPlaying: !prev.isPlaying }))
  }, [])

  const setSimulationSpeed = useCallback((speed: number) => {
    setState((prev) => ({ ...prev, speed }))
  }, [])

  const setWeather = useCallback((weather: 'clear' | 'rain' | 'snow' | 'fog') => {
    setState((prev) => ({ ...prev, weather }))
    updateSimulation()
  }, [updateSimulation])

  const setTime = useCallback(
    (time: number) => {
      setState((prev) => ({ ...prev, currentTime: time }))
      updateSimulation(time)
    },
    [updateSimulation]
  )

  return {
    ...state,
    togglePlayPause,
    setSimulationSpeed,
    setWeather,
    setTime,
    updateSimulation,
  }
}
