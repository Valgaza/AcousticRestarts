'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Clock } from 'lucide-react'
import { useTrafficSimulation, GridCell } from '@/hooks/useTrafficSimulation'
import ContextPanel from './ContextPanel'
import TimeScrubber from './TimeScrubber'

interface ViewportProps {
  state: ReturnType<typeof useTrafficSimulation>
}

export default function Viewport({ state }: ViewportProps) {
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null)

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
        <div className="flex items-center gap-3">
          <Clock className="w-5 h-5 text-slate-400" />
          <div>
            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Current Time</p>
            <p className="text-lg font-mono font-bold text-slate-100">{formatTime(state.currentTime)}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Simulation</p>
          <p className={`text-sm font-mono font-bold ${state.isPlaying ? 'text-emerald-400' : 'text-slate-400'}`}>
            {state.isPlaying ? 'ACTIVE' : 'PAUSED'}
          </p>
        </div>
      </div>

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

          {/* Traffic Grid */}
          <div className="absolute inset-0 grid gap-0 grid-cols-25 grid-rows-25 border border-slate-700">
            {state.grid.flat().map((cell) => (
              <motion.button
                key={`${cell.x}-${cell.y}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setSelectedCell(cell)
                }}
                className={`border border-slate-700 transition-all hover:border-slate-500 cursor-pointer relative group ${getCongestionColor(cell.congestionLevel)}`}
                initial={{ opacity: 0.7 }}
                animate={{
                  opacity: 0.3 + cell.congestionLevel * 0.7,
                }}
                transition={{ duration: 0.3 }}
                title={`Cell ${cell.x},${cell.y}: ${Math.round(cell.congestionLevel * 100)}%`}
              >
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-slate-100/10 transition-opacity" />
              </motion.button>
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
      <TimeScrubber state={state} />
    </div>
  )
}
