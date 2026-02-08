'use client'

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Clock, RefreshCw, Wifi, WifiOff, Timer, Play, Pause } from 'lucide-react'
import { useTrafficSimulation, GridCell } from '@/hooks/useTrafficSimulation'
import type { useLiveTraffic } from '@/hooks/useLiveTraffic'
import type { useGridTraffic } from '@/hooks/useGridTraffic'
import ContextPanel from './ContextPanel'
import TimeScrubber from './TimeScrubber'
import { Button } from './ui/button'

interface ViewportProps {
  state: ReturnType<typeof useTrafficSimulation>
  liveTraffic: ReturnType<typeof useLiveTraffic>
  gridTraffic: ReturnType<typeof useGridTraffic>
  useLiveData?: boolean
}

export default function Viewport({ state, liveTraffic, gridTraffic, useLiveData = true }: ViewportProps) {
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null)

  // Use grid traffic data
  const activeGrid = gridTraffic.grid
  const isLoading = gridTraffic.isLoading
  const lastUpdate = gridTraffic.lastUpdate
  const gridKey = `grid-frame-${gridTraffic.currentFrameIndex}`
  const currentFrame = gridTraffic.frames[gridTraffic.currentFrameIndex]

  // Debug: Log when grid changes
  React.useEffect(() => {
    const nonZeroCells = gridTraffic.grid.flat().filter(c => c.congestionLevel > 0).length
    const avgCongestion = gridTraffic.grid.flat().reduce((sum, c) => sum + c.congestionLevel, 0) / (25*25)
    console.log(`🔄 Grid frame ${gridTraffic.currentFrameIndex}/${gridTraffic.frames.length}, Active cells: ${nonZeroCells}, Avg: ${avgCongestion.toFixed(3)}`)
  }, [gridTraffic.currentFrameIndex, gridTraffic.grid, gridTraffic.frames.length])

  const getCongestionColor = (level: number) => {
    if (level < 0.3) return 'bg-emerald-500' // Low
    if (level < 0.6) return 'bg-yellow-500' // Moderate
    if (level < 0.8) return 'bg-rose-500' // Heavy
    return 'bg-indigo-500' // Gridlock
  }

  const formatTime = (hour: number) => {
    const h = Math.floor(hour)
    const m = Math.round((hour - h) * 60)
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
  }

  return (
    <div className="flex-1 flex flex-col bg-gradient-to-b from-slate-950 to-slate-900 relative overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-700 flex items-center justify-between bg-slate-900/50 backdrop-blur">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <Clock className="w-5 h-5 text-slate-400" />
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Current Time</p>
              <p className="text-lg font-mono font-bold text-slate-100">{formatTime(state.currentTime)}</p>
            </div>
          </div>

          {useLiveData && (
            <>
              <div className="h-8 w-px bg-slate-700" />
              <div className="flex items-center gap-3">
                {liveTraffic.autoRefresh ? (
                  <Wifi className="w-5 h-5 text-emerald-400" />
                ) : (
                  <WifiOff className="w-5 h-5 text-slate-400" />
                )}
                <div>
                  <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Live Data</p>
                  <p className={`text-sm font-mono font-bold ${liveTraffic.autoRefresh ? 'text-emerald-400' : 'text-slate-400'}`}>
                    {liveTraffic.autoRefresh ? 'STREAMING' : 'PAUSED'}
                  </p>
                </div>
              </div>
              {lastUpdate && (
                <div className="text-xs text-slate-500" suppressHydrationWarning>
                  Updated: {lastUpdate.toLocaleTimeString()}
                </div>
              )}
            </>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex gap-2">
            {/* Frame Navigation */}
            <div className="flex items-center gap-2 border border-slate-700 rounded-md px-3 py-1">
              <Timer className="w-4 h-4 text-slate-400" />
              <span className="text-sm font-mono text-slate-100">
                Frame {gridTraffic.currentFrameIndex + 1}/{gridTraffic.frames.length}
              </span>
            </div>

            <Button
              size="sm"
              variant="outline"
              onClick={gridTraffic.manualRefresh}
              disabled={isLoading}
              className="gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
          
          <div className="text-right">
            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">
              Mode
            </p>
            <p className="text-sm font-mono font-bold text-emerald-400">
              GRID DATA
            </p>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {gridTraffic.error && (
        <div className="px-6 py-3 bg-rose-500/10 border-b border-rose-500/20 text-rose-400 text-sm">
          ⚠️ Error: {gridTraffic.error}
        </div>
      )}

      {/* Map Grid Container */}
      <div 
        className="flex-1 flex items-center justify-center p-8 overflow-hidden"
        onClick={() => selectedCell && setSelectedCell(null)}
      >
        <div className="relative w-full h-full max-w-2xl max-h-2xl">
          {/* Grid Background */}
          <svg
            className="absolute inset-0 w-full h-full opacity-10"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            <defs>
              <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="100" height="100" fill="url(#grid)" stroke="currentColor" strokeWidth="1" />
          </svg>

          {/* Road Network Image Background */}
          <div 
            className="absolute inset-0 bg-cover bg-center opacity-100"
            style={{
              backgroundImage: 'url(/roadNetwork.jpeg)',
              backgroundSize: '100% 100%',
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'center'
            }}
          />

          {/* Traffic Grid */}
          <div className="absolute inset-0 grid gap-0 grid-cols-25 grid-rows-25 border border-slate-700" key={gridKey}>
            {activeGrid.flat().map((cell) => (
              <motion.div
                key={`${gridKey}-${cell.x}-${cell.y}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedCell(cell)
                }}
                className={`border border-slate-700 transition-all hover:border-slate-500 cursor-pointer relative group ${getCongestionColor(cell.congestionLevel)}`}
                animate={{
                  opacity: 0.3 + cell.congestionLevel * 0.7,
                }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                title={`Cell ${cell.x},${cell.y}: ${Math.round(cell.congestionLevel * 100)}% congestion | Frame: ${gridTraffic.currentFrameIndex + 1}`}
              >
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-slate-100/10 transition-opacity" />
              </motion.div>
            ))}
          </div>

          {/* Overlay Info */}
          {selectedCell && (
            <ContextPanel
              cell={selectedCell}
              onClose={() => setSelectedCell(null)}
            />
          )}
        </div>
      </div>

      {/* Time Scrubber */}
      <TimeScrubber gridTraffic={gridTraffic} />
    </div>
  )
}
