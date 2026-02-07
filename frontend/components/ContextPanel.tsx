'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { GridCell } from '@/hooks/useTrafficSimulation'

interface ContextPanelProps {
  cell: GridCell
  onClose: () => void
}

export default function ContextPanel({ cell, onClose }: ContextPanelProps) {
  const congestionPercent = Math.round(cell.congestionLevel * 100)
  const predictedPercent = Math.round(cell.predictedCongestion * 100)

  const getCongestionLabel = (level: number) => {
    if (level < 0.3) return 'Low'
    if (level < 0.6) return 'Moderate'
    if (level < 0.8) return 'Heavy'
    return 'Gridlock'
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 20 }}
        transition={{ type: 'spring', damping: 20, stiffness: 300 }}
        onClick={(e) => e.stopPropagation()}
        className="absolute bottom-6 left-6 z-50"
      >
        <Card className="bg-slate-800 border border-slate-600 shadow-2xl w-80">
          <div className="p-4 space-y-4">
            {/* Header */}
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Node ID</p>
                <p className="text-lg font-mono font-bold text-slate-100">
                  {cell.x}-{cell.y}
                </p>
              </div>
              <Button
                onClick={onClose}
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 text-slate-400 hover:text-slate-100"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            {/* Current Speed */}
            <div className="bg-slate-900 p-3 border border-slate-700">
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Current Speed</p>
              <p className="text-2xl font-mono font-bold text-emerald-400">
                {Math.round(cell.speed)} km/h
              </p>
            </div>

            {/* Congestion Level */}
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-slate-900 p-3 border border-slate-700">
                <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Congestion</p>
                <p className="text-xl font-mono font-bold text-rose-400">{congestionPercent}%</p>
                <p className="text-xs font-mono text-slate-500 mt-1">{getCongestionLabel(cell.congestionLevel)}</p>
              </div>

              <div className="bg-slate-900 p-3 border border-slate-700">
                <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">Predicted</p>
                <p className="text-xl font-mono font-bold text-yellow-400">{predictedPercent}%</p>
                <p className="text-xs font-mono text-slate-500 mt-1">+1 Hour</p>
              </div>
            </div>

            {/* Mini Chart */}
            <div className="bg-slate-900 p-3 border border-slate-700">
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">Trend</p>
              <div className="flex items-end gap-1 h-12">
                {Array.from({ length: 12 }).map((_, i) => {
                  const value = Math.sin((i / 12) * Math.PI) * 0.5 + 0.3 + Math.random() * 0.1
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-slate-600 rounded-t-sm"
                      style={{
                        height: `${value * 100}%`,
                        backgroundColor: value > 0.7 ? '#f43f5e' : value > 0.5 ? '#eab308' : '#10b981',
                      }}
                    />
                  )
                })}
              </div>
            </div>

            {/* Close Note */}
            <p className="text-xs text-slate-500 text-center font-mono">
              Click anywhere to close
            </p>
          </div>
        </Card>
      </motion.div>
    </AnimatePresence>
  )
}
