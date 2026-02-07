'use client'

import { Play, Pause, Cloud, CloudRain, CloudSnow, CloudFog, Activity, Radio } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTrafficSimulation, RouteMetric } from '@/hooks/useTrafficSimulation'
import type { useLiveTraffic } from '@/hooks/useLiveTraffic'
import RouteSparkline from './RouteSparkline'

interface ControlSidebarProps {
  state: ReturnType<typeof useTrafficSimulation>
  liveTraffic: ReturnType<typeof useLiveTraffic>
  useLiveData?: boolean
  onToggleLive?: (value: boolean) => void
}

export default function ControlSidebar({ state, liveTraffic, useLiveData = false, onToggleLive }: ControlSidebarProps) {
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

      {/* Live Data Toggle */}
      {onToggleLive && (
        <Card className="bg-slate-800 border-slate-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className={`w-4 h-4 ${useLiveData ? 'text-emerald-400' : 'text-slate-400'}`} />
              <div>
                <p className="text-sm font-mono text-slate-200">Live API Data</p>
                <p className="text-xs text-slate-400">Real-time predictions</p>
              </div>
            </div>
            <Switch
              checked={useLiveData}
              onCheckedChange={onToggleLive}
            />
          </div>
        </Card>
      )}

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

      {/* Route Metrics */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-slate-400" />
          <p className="text-xs font-mono text-slate-300 uppercase tracking-wider">Critical Arteries</p>
        </div>
        <div className="space-y-3">
          {state.routes.map((route) => (
            <RouteMetricCard key={route.id} route={route} />
          ))}
        </div>
      </div>
    </div>
  )
}

function RouteMetricCard({ route }: { route: RouteMetric }) {
  const congestionPercent = Math.round(((route.freeFlowSpeed - route.currentSpeed) / route.freeFlowSpeed) * 100)
  const isCongested = congestionPercent > 40

  return (
    <Card className="bg-slate-800 border-slate-700 p-3 hover:border-slate-600 cursor-pointer transition-colors">
      <div className="space-y-2">
        <p className="text-xs font-mono text-slate-100 font-semibold">{route.name}</p>
        <div className="flex justify-between items-center gap-2">
          <div>
            <p className="text-xs text-slate-400 font-mono">Speed</p>
            <p className={`text-sm font-mono font-bold ${isCongested ? 'text-rose-400' : 'text-emerald-400'}`}>
              {route.currentSpeed} km/h
            </p>
          </div>
          <div className="flex-1 h-8">
            <RouteSparkline data={route.sparklineData} />
          </div>
        </div>
        <div className="w-full bg-slate-700 rounded-none h-1 overflow-hidden">
          <div
            className={`h-full transition-all ${isCongested ? 'bg-rose-500' : 'bg-emerald-500'}`}
            style={{
              width: `${congestionPercent}%`,
            }}
          />
        </div>
      </div>
    </Card>
  )
}
