'use client'

import { Play, Pause, Cloud, CloudRain, CloudSnow, CloudFog, Map } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import type { useLiveTraffic } from '@/hooks/useLiveTraffic'

interface ControlSidebarProps {
  state: ReturnType<typeof useTrafficSimulation>
  liveTraffic: ReturnType<typeof useLiveTraffic>
  useLiveData?: boolean
}

export default function ControlSidebar({ state, liveTraffic, useLiveData = true }: ControlSidebarProps) {
  const weatherIcons = {
    clear: <Cloud className="w-4 h-4" />,
    rain: <CloudRain className="w-4 h-4" />,
    snow: <CloudSnow className="w-4 h-4" />,
    fog: <CloudFog className="w-4 h-4" />,
  }

  const weatherLabels = {
    clear: 'Clear',
    rain: 'Rain',
    snow: 'Snow',
    fog: 'Fog',
  }

  return (
    <div className="w-80 bg-slate-900 border-r border-slate-700 flex flex-col p-4 gap-4 overflow-y-auto">
      {/* Header */}
      <div className="mb-2">
        <h1 className="text-xl font-bold font-mono text-slate-100 tracking-wider">TRAFFIC.OS</h1>
        <p className="text-xs text-slate-400 font-mono mt-1">Urban Congestion Management</p>
      </div>

      {/* Simulation Controls */}
      <Card className="bg-slate-800 border-slate-700 p-4">
        <div className="space-y-4">
          {/* Play/Pause */}
          <div>
            <p className="text-xs font-mono text-slate-300 mb-2 uppercase tracking-wider">
              {useLiveData ? 'Time Horizons' : 'Simulation'}
            </p>
            <Button
              onClick={useLiveData ? liveTraffic.togglePlayPause : state.togglePlayPause}
              className="w-full bg-slate-700 hover:bg-slate-600 text-slate-100 font-mono text-sm border border-slate-600"
              variant="outline"
            >
              {(useLiveData ? liveTraffic.isPlaying : state.isPlaying) ? (
                <>
                  <Pause className="w-4 h-4 mr-2" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Play
                </>
              )}
            </Button>
            {useLiveData && (
              <p className="text-xs text-slate-400 mt-2">
                {liveTraffic.isPlaying ? 'Cycling through t+1h to t+6h' : 'Paused at ' + liveTraffic.selectedHorizon}
              </p>
            )}
          </div>

          {/* Speed Control */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <p className="text-xs font-mono text-slate-300 uppercase tracking-wider">Speed</p>
              <span className="text-sm font-mono text-slate-200 bg-slate-700 px-2 py-1">
                {useLiveData ? `${liveTraffic.speed}s` : `${state.speed}x`}
              </span>
            </div>
            <div className="flex gap-1">
              {useLiveData ? (
                /* Seconds per horizon for live mode */
                [2, 3, 5].map((speed) => (
                  <Button
                    key={speed}
                    onClick={() => liveTraffic.setSpeed(speed)}
                    className={`flex-1 text-xs font-mono ${
                      liveTraffic.speed === speed
                        ? 'bg-slate-600 text-slate-100 border-slate-500'
                        : 'bg-slate-700 hover:bg-slate-600 text-slate-300 border-slate-600'
                    }`}
                    variant="outline"
                  >
                    {speed}s
                  </Button>
                ))
              ) : (
                /* Multiplier for simulation mode */
                [1, 5, 10].map((speed) => (
                  <Button
                    key={speed}
                    onClick={() => state.setSimulationSpeed(speed)}
                    className={`flex-1 text-xs font-mono ${
                      state.speed === speed
                        ? 'bg-slate-600 text-slate-100 border-slate-500'
                        : 'bg-slate-700 hover:bg-slate-600 text-slate-300 border-slate-600'
                    }`}
                    variant="outline"
                  >
                    {speed}x
                  </Button>
                ))
              )}
            </div>
          </div>

          {/* Weather Control */}
          <div>
            <p className="text-xs font-mono text-slate-300 mb-2 uppercase tracking-wider">
              Weather
            </p>
            <Select value={state.weather} onValueChange={(v: any) => state.setWeather(v)}>
              <SelectTrigger className="bg-slate-700 border-slate-600 text-slate-100 font-mono text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-700">
                {Object.entries(weatherLabels).map(([key, label]) => (
                  <SelectItem key={key} value={key} className="font-mono text-xs">
                    {label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </Card>

      {/* Map Legend */}
      <Card className="bg-slate-800 border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-3">
          <Map className="w-4 h-4 text-slate-400" />
          <p className="text-xs font-mono text-slate-300 uppercase tracking-wider">Map Legend</p>
        </div>
        <div className="space-y-3">
          {/* Road Types */}
          <div className="space-y-2">
            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Road Types</p>
            <div className="space-y-1.5">
              <LegendItem color="bg-red-500" label="Highway" count={12} />
              <LegendItem color="bg-cyan-400" label="Arterial" count={106} />
              <LegendItem color="bg-emerald-500" label="Residential" count={266} />
              <div className="flex items-center gap-2">
                <div className="w-8 h-0.5 border-t-2 border-dashed border-yellow-500" />
                <span className="text-xs font-mono text-slate-200">Ramp</span>
                <span className="text-xs font-mono text-slate-400">({16})</span>
              </div>
            </div>
          </div>

          {/* Nodes */}
          <div className="space-y-2">
            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Nodes</p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-pink-400 border-2 border-pink-300" />
                <span className="text-xs font-mono text-slate-200">CBD node</span>
                <span className="text-xs font-mono text-slate-400">(&lt;500m)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-cyan-400 border-2 border-cyan-300" />
                <span className="text-xs font-mono text-slate-200">Middle node</span>
                <span className="text-xs font-mono text-slate-400">(500m-1km)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-slate-400 border-2 border-slate-300" />
                <span className="text-xs font-mono text-slate-200">Suburban node</span>
                <span className="text-xs font-mono text-slate-400">(&gt;1km)</span>
              </div>
            </div>
          </div>

          {/* Zones */}
          <div className="space-y-2">
            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">Zones</p>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <div className="w-4 h-3 bg-purple-500/30 border border-purple-400" />
                <span className="text-xs font-mono text-slate-200">CBD zone</span>
                <span className="text-xs font-mono text-slate-400">(&lt;500m)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-3 bg-blue-500/20 border border-blue-400" />
                <span className="text-xs font-mono text-slate-200">Suburban boundary</span>
                <span className="text-xs font-mono text-slate-400">(1km)</span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

function LegendItem({ color, label, count }: { color: string; label: string; count: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`w-8 h-0.5 ${color}`} />
      <span className="text-xs font-mono text-slate-200">{label}</span>
      <span className="text-xs font-mono text-slate-400">({count})</span>
    </div>
  )
}
