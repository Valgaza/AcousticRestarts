'use client'

import { useGridTraffic } from '@/hooks/useGridTraffic'

interface TimeScrubberProps {
  gridTraffic: ReturnType<typeof useGridTraffic>
}

export default function TimeScrubber({ gridTraffic }: TimeScrubberProps) {
  const formatDateTime = (dateTimeStr: string) => {
    try {
      const date = new Date(dateTimeStr)
      const hours = date.getHours()
      const minutes = date.getMinutes()
      const month = date.getMonth() + 1
      const day = date.getDate()
      const year = date.getFullYear() % 100
      return `${String(day).padStart(2, '0')},${String(month).padStart(2, '0')},${String(year).padStart(2, '0')} ${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`
    } catch {
      return dateTimeStr
    }
  }

  const currentFrame = gridTraffic.frames[gridTraffic.currentFrameIndex]
  const displayTime = currentFrame ? formatDateTime(currentFrame.DateTime) : '--:--'
  
  // Generate time labels from available frames
  const getTimeLabels = () => {
    if (gridTraffic.frames.length === 0) return []
    
    const labelCount = Math.min(6, gridTraffic.frames.length)
    const step = Math.max(1, Math.floor(gridTraffic.frames.length / (labelCount - 1)))
    
    const labels = []
    for (let i = 0; i < labelCount; i++) {
      const index = Math.min(i * step, gridTraffic.frames.length - 1)
      const frame = gridTraffic.frames[index]
      if (frame) {
        try {
          const date = new Date(frame.DateTime)
          const hours = date.getHours()
          const minutes = date.getMinutes()
          labels.push(`${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`)
        } catch {
          labels.push('--:--')
        }
      }
    }
    return labels
  }

  const timeLabels = getTimeLabels()
  const maxFrames = Math.max(0, gridTraffic.frames.length - 1)

  return (
    <div className="bg-slate-900/80 backdrop-blur border-t border-slate-700 p-4">
      <div className="max-w-full">
        <div className="flex justify-between items-center mb-3">
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Timeline</p>
          <p className="text-sm font-mono font-bold text-slate-100">{displayTime}</p>
        </div>

        {/* Scrubber Track */}
        <div className="relative h-12 flex items-center">
          {/* Background Track */}
          <div className="absolute inset-x-0 h-2 bg-slate-800 border border-slate-700" />

          {/* Gradient overlay showing congestion pattern */}
          <div className="absolute inset-x-0 h-2 flex">
            {Array.from({ length: 100 }).map((_, i) => {
              if (gridTraffic.frames.length === 0) {
                return <div key={i} className="flex-1 h-2 bg-slate-700" />
              }
              
              // Map this segment to a frame in the timeline
              const framePosition = (i / 100) * (gridTraffic.frames.length - 1)
              const frameIndex = Math.floor(framePosition)
              const nextFrameIndex = Math.min(frameIndex + 1, gridTraffic.frames.length - 1)
              
              const frame = gridTraffic.frames[frameIndex]
              const nextFrame = gridTraffic.frames[nextFrameIndex]
              
              // Calculate average congestion for current frame
              const getAvgCongestion = (f: typeof frame) => {
                if (!f || f.cells.length === 0) return 0.3
                return f.cells.reduce((sum, cell) => sum + cell.predicted_congestion_level, 0) / f.cells.length
              }
              
              const congestion1 = getAvgCongestion(frame)
              const congestion2 = getAvgCongestion(nextFrame)
              
              // Interpolate between frames for smooth gradient
              const alpha = framePosition - frameIndex
              const avgCongestion = congestion1 * (1 - alpha) + congestion2 * alpha
              
              // Map congestion to colors based on actual data range (0-1 scale)
              // Low: 0-0.25 (green), Moderate: 0.25-0.50 (yellow), High: 0.50+ (red)
              const color = avgCongestion > 0.50 ? '#f43f5e' : avgCongestion > 0.25 ? '#eab308' : '#10b981'
              return (
                <div
                  key={i}
                  className="flex-1 h-2"
                  style={{ backgroundColor: color }}
                />
              )
            })}
          </div>

          {/* Slider Input */}
          <input
            type="range"
            min="0"
            max={maxFrames}
            step="1"
            value={gridTraffic.currentFrameIndex}
            onChange={(e) => gridTraffic.setFrameIndex(parseInt(e.currentTarget.value))}
            disabled={gridTraffic.frames.length === 0}
            className="absolute inset-x-0 h-2 w-full appearance-none bg-transparent cursor-pointer z-20 slider"
            style={{
              WebkitAppearance: 'slider-horizontal',
            }}
          />

          {/* Time Labels */}
          <div className="absolute inset-x-0 flex mt-6 text-xs font-mono text-slate-500 px-2">
            {timeLabels.map((time, i) => (
              <div
                key={time}
                className="flex-1 text-center"
                style={{ marginLeft: i === 0 ? '0' : 'auto' }}
              >
                {time}
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 24px;
          height: 24px;
          border-radius: 0;
          background: #0ea5e9;
          border: 2px solid #0369a1;
          cursor: pointer;
          box-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
        }

        .slider::-moz-range-thumb {
          width: 24px;
          height: 24px;
          border-radius: 0;
          background: #0ea5e9;
          border: 2px solid #0369a1;
          cursor: pointer;
          box-shadow: 0 0 10px rgba(14, 165, 233, 0.5);
        }
      `}</style>
    </div>
  )
}
