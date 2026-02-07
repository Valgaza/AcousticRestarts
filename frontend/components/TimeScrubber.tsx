'use client'

import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'

interface TimeScrubberProps {
  state: ReturnType<typeof useTrafficSimulation>
}

export default function TimeScrubber({ state }: TimeScrubberProps) {
  const formatTime = (hour: number) => {
    const h = Math.floor(hour)
    const m = Math.round((hour - h) * 60)
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
  }

  const timeLabels = ['06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00']

  return (
    <div className="bg-slate-900/80 backdrop-blur border-t border-slate-700 p-4">
      <div className="max-w-full">
        <div className="flex justify-between items-center mb-3">
          <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Timeline</p>
          <p className="text-sm font-mono font-bold text-slate-100">{formatTime(state.currentTime)}</p>
        </div>

        {/* Scrubber Track */}
        <div className="relative h-12 flex items-center">
          {/* Background Track */}
          <div className="absolute inset-x-0 h-2 bg-slate-800 border border-slate-700" />

          {/* Gradient overlay showing congestion pattern */}
          <div className="absolute inset-x-0 h-2 flex">
            {Array.from({ length: 100 }).map((_, i) => {
              const hour = 6 + (i / 100) * 16 // 6:00 to 22:00
              const rushHour = Math.sin(((hour - 6) * Math.PI) / 6) * 0.5 + 0.5
              const color = rushHour > 0.7 ? '#f43f5e' : rushHour > 0.5 ? '#eab308' : '#10b981'
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
            max="24"
            step="0.1"
            value={state.currentTime}
            onChange={(e) => state.setTime(parseFloat(e.currentTarget.value))}
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
