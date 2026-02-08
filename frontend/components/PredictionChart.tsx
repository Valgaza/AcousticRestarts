'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { useGridTraffic } from '@/hooks/useGridTraffic'

interface PredictionChartProps {
  gridTraffic: ReturnType<typeof useGridTraffic>
}

export default function PredictionChart({ gridTraffic }: PredictionChartProps) {
  // Generate data from actual grid frames
  const data = gridTraffic.frames.map((frame, index) => {
    // Calculate average congestion for the frame
    const avgCongestion = frame.cells.length > 0
      ? frame.cells.reduce((sum, cell) => sum + cell.predicted_congestion_level, 0) / frame.cells.length
      : 0
    
    // Calculate max congestion (hotspot)
    const maxCongestion = frame.cells.length > 0
      ? Math.max(...frame.cells.map(cell => cell.predicted_congestion_level))
      : 0
    
    // Format time from DateTime
    let timeLabel = `T+${index + 1}`
    try {
      const date = new Date(frame.DateTime)
      const hours = date.getHours()
      const minutes = date.getMinutes()
      timeLabel = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`
    } catch {}
    
    return {
      time: timeLabel,
      average: Math.round(avgCongestion * 100),
      peak: Math.round(maxCongestion * 100),
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
          label={{ value: 'Congestion %', angle: -90, position: 'insideLeft', style: { fill: '#94a3b8', fontSize: '11px' } }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid #475569',
            borderRadius: '0',
          }}
          labelStyle={{ color: '#e2e8f0', fontFamily: 'monospace', fontSize: '12px' }}
          formatter={(value: number) => `${value}%`}
        />
        <Legend 
          wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace' }}
          iconType="line"
        />
        <Line
          type="monotone"
          dataKey="average"
          stroke="#10b981"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Average"
          isAnimationActive={true}
          animationDuration={500}
        />
        <Line
          type="monotone"
          dataKey="peak"
          stroke="#f43f5e"
          strokeWidth={2}
          dot={{ r: 3 }}
          name="Peak"
          isAnimationActive={true}
          animationDuration={500}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
