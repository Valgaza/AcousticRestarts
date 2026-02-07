'use client'

import { useState } from 'react'
import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import { useLiveTraffic } from '@/hooks/useLiveTraffic'
import ControlSidebar from '@/components/ControlSidebar'
import Viewport from '@/components/Viewport'
import AnalyticsPanel from '@/components/AnalyticsPanel'

export default function Dashboard() {
  const state = useTrafficSimulation()
  const liveTraffic = useLiveTraffic()
  const [useLiveData, setUseLiveData] = useState(false)

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Left Sidebar */}
      <ControlSidebar state={state} liveTraffic={liveTraffic} useLiveData={useLiveData} onToggleLive={setUseLiveData} />

      {/* Main Viewport */}
      <Viewport state={state} liveTraffic={liveTraffic} useLiveData={useLiveData} />

      {/* Right Analytics Panel */}
      <AnalyticsPanel state={state} liveTraffic={liveTraffic} useLiveData={useLiveData} />
    </div>
  )
}
