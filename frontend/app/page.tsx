'use client'

import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import ControlSidebar from '@/components/ControlSidebar'
import Viewport from '@/components/Viewport'
import AnalyticsPanel from '@/components/AnalyticsPanel'

export default function Dashboard() {
  const state = useTrafficSimulation()

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Left Sidebar */}
      <ControlSidebar state={state} />

      {/* Main Viewport */}
      <Viewport state={state} />

      {/* Right Analytics Panel */}
      <AnalyticsPanel state={state} />
    </div>
  )
}
