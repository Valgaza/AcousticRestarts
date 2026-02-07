'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'

interface PredictionChartProps {
  state: ReturnType<typeof useTrafficSimulation>
}

export default function PredictionChart({ state }: PredictionChartProps) {
  // Generate mock actual vs predicted data
  const data = Array.from({ length: 12 }).map((_, i) => {
    const hour = state.currentTime + (i - 5)
    const rushHour = Math.sin(((hour - 6) * Math.PI) / 6) * 0.5 + 0.5
    return {
      time: `${Math.floor(hour)}:00`,
      actual: Math.max(0, Math.min(100, rushHour * 100 + Math.random() * 10)),
      predicted: Math.max(0, Math.min(100, rushHour * 100 + 5)),
    }
  })

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
        <XAxis
          dataKey="time"
          stroke="#64748b"
          style={{ fontSize: '11px', fontFamily: 'monospace' }}
          tick={{ fill: '#94a3b8' }}
        />
        <YAxis
          stroke="#64748b"
          style={{ fontSize: '11px', fontFamily: 'monospace' }}
          tick={{ fill: '#94a3b8' }}
          domain={[0, 100]}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid #475569',
            borderRadius: '0',
          }}
          labelStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: '12px' }}
          formatter={(value: number) => `${Math.round(value)}%`}
        />
        <Line
          type="monotone"
          dataKey="actual"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          name="Actual"
          isAnimationActive={true}
          animationDuration={500}
        />
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#f43f5e"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={false}
          name="Predicted"
          isAnimationActive={true}
          animationDuration={500}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
