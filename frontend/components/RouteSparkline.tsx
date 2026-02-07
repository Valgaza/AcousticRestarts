'use client'

import { LineChart, Line, ResponsiveContainer } from 'recharts'

interface RouteSparklineProps {
  data: number[]
}

export default function RouteSparkline({ data }: RouteSparklineProps) {
  const chartData = data.map((value, index) => ({
    index,
    value,
  }))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
        <Line
          type="monotone"
          dataKey="value"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
