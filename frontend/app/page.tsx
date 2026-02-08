'use client'

import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import { useLiveTraffic } from '@/hooks/useLiveTraffic'
import { useGridTraffic } from '@/hooks/useGridTraffic'
import ControlSidebar from '@/components/ControlSidebar'
import Viewport from '@/components/Viewport'
import AnalyticsPanel from '@/components/AnalyticsPanel'

export default function Dashboard() {
  const state = useTrafficSimulation()
  const liveTraffic = useLiveTraffic()
  const gridTraffic = useGridTraffic()

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Left Sidebar */}
      <ControlSidebar state={state} liveTraffic={liveTraffic} gridTraffic={gridTraffic} useLiveData={true} />

      {/* Main Viewport */}
      <Viewport state={state} liveTraffic={liveTraffic} gridTraffic={gridTraffic} useLiveData={true} />

      {/* Right Analytics Panel */}
      <AnalyticsPanel state={state} liveTraffic={liveTraffic} gridTraffic={gridTraffic} useLiveData={true} />
    </div>
  )
}
