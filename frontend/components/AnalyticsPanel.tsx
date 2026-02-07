'use client'

import { useState, useMemo, useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown, ChevronUp, TrendingUp, AlertTriangle, Upload, CheckCircle, XCircle } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import type { useLiveTraffic } from '@/hooks/useLiveTraffic'
import PredictionChart from './PredictionChart'
import { uploadCsvFile } from '@/lib/trafficApi'

interface AnalyticsPanelProps {
  state: ReturnType<typeof useTrafficSimulation>
  liveTraffic: ReturnType<typeof useLiveTraffic>
  useLiveData?: boolean
}

export default function AnalyticsPanel({ state, liveTraffic, useLiveData = true }: AnalyticsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [uploadMessage, setUploadMessage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadStatus('uploading')
    setUploadMessage('Uploading...')

    try {
      const result = await uploadCsvFile(file)
      setUploadStatus('success')
      setUploadMessage(`File uploaded: ${result.filename} (${(result.size_bytes / 1024).toFixed(2)} KB)`)
      
      // Clear success message after 5 seconds
      setTimeout(() => {
        setUploadStatus('idle')
        setUploadMessage('')
      }, 5000)
    } catch (error) {
      setUploadStatus('error')
      setUploadMessage(error instanceof Error ? error.message : 'Upload failed')
      
      // Clear error message after 5 seconds
      setTimeout(() => {
        setUploadStatus('idle')
        setUploadMessage('')
      }, 5000)
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Calculate live metrics from grid data
  const liveMetrics = useMemo(() => {
    if (!useLiveData || liveTraffic.grid.length === 0) return null

    const cells = liveTraffic.grid.flat()
    const congestionLevels = cells.map(c => c.congestionLevel).filter(c => c > 0)
    
    const avgCongestion = congestionLevels.length > 0 
      ? congestionLevels.reduce((a, b) => a + b, 0) / congestionLevels.length 
      : 0
    
    const avgSpeed = Math.round(60 * (1 - avgCongestion * 0.7))
    const bottlenecks = cells.filter(c => c.congestionLevel > 0.7).length
    
    return {
      avgCitySpeed: avgSpeed,
      activeBottlenecks: bottlenecks
    }
  }, [useLiveData, liveTraffic.grid])

  // Choose data source
  const metrics = useLiveData && liveMetrics ? liveMetrics : state

  const stats = [
    {
      label: 'Avg City Speed',
      value: `${metrics.avgCitySpeed} km/h`,
      icon: TrendingUp,
      color: 'text-emerald-400',
      change: useLiveData ? `${liveTraffic.selectedHorizon} forecast` : '+5% from baseline',
    },
    {
      label: 'Active Bottlenecks',
      value: metrics.activeBottlenecks.toString(),
      icon: AlertTriangle,
      color: 'text-rose-400',
      change: 'Critical areas',
    },
  ]

  return (
    <div className="w-96 bg-slate-900 border-l border-slate-700 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="border-b border-slate-700 p-4 flex items-center justify-between">
        <h2 className="text-sm font-mono font-bold text-slate-100 uppercase tracking-wider">Analytics</h2>
        <Button
          onClick={() => setIsExpanded(!isExpanded)}
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 text-slate-400 hover:text-slate-100"
        >
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </Button>
      </div>

      {/* Content */}
      {isExpanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="flex-1 overflow-y-auto"
        >
          <div className="p-4 space-y-4">
            {/* System Status */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Current System Status
              </p>
              <div className="space-y-2">
                {stats.map((stat, i) => {
                  const Icon = stat.icon
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                    >
                      <Card className="bg-slate-800 border-slate-700 p-3">
                        <div className="flex items-start gap-3">
                          <Icon className={`w-5 h-5 mt-1 flex-shrink-0 ${stat.color}`} />
                          <div className="flex-1 min-w-0">
                            <p className="text-xs font-mono text-slate-400 uppercase tracking-wider">
                              {stat.label}
                            </p>
                            <p className={`text-lg font-mono font-bold ${stat.color} mt-1 break-words`}>
                              {stat.value}
                            </p>
                            <p className="text-xs text-slate-500 font-mono mt-1">{stat.change}</p>
                          </div>
                        </div>
                      </Card>
                    </motion.div>
                  )
                })}
              </div>
            </div>

            {/* CSV Upload Section */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Upload Data for ML Model
              </p>
              <Card className="bg-slate-800 border-slate-700 p-4">
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-slate-300">
                    <Upload className="w-4 h-4" />
                    <span className="text-xs font-mono">CSV File Upload</span>
                  </div>
                  
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="csv-upload"
                  />
                  
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadStatus === 'uploading'}
                    className="w-full bg-slate-700 hover:bg-slate-600 text-slate-100 border border-slate-600"
                    size="sm"
                  >
                    {uploadStatus === 'uploading' ? (
                      <>
                        <div className="w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin mr-2" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload className="w-3 h-3 mr-2" />
                        Select CSV File
                      </>
                    )}
                  </Button>

                  {uploadMessage && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex items-start gap-2 p-2 border text-xs font-mono ${
                        uploadStatus === 'success'
                          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                          : uploadStatus === 'error'
                          ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                          : 'bg-slate-700 border-slate-600 text-slate-300'
                      }`}
                    >
                      {uploadStatus === 'success' && <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />}
                      {uploadStatus === 'error' && <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />}
                      <span className="flex-1">{uploadMessage}</span>
                    </motion.div>
                  )}

                  <p className="text-xs text-slate-500 font-mono">
                    Only .csv files accepted
                  </p>
                </div>
              </Card>
            </div>

            {/* Prediction Chart */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Network Congestion Forecast
              </p>
              <Card className="bg-slate-800 border-slate-700 p-4">
                <PredictionChart state={state} />
              </Card>
            </div>

            {/* Status Indicators */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">Status</p>
              <div className="space-y-2">
                <div className="flex items-center gap-2 p-2 bg-slate-800 border border-slate-700">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-mono text-slate-200">System Operational</span>
                </div>
                <div className="flex items-center gap-2 p-2 bg-slate-800 border border-slate-700">
                  <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                  <span className="text-xs font-mono text-slate-200">{state.activeBottlenecks} Alert(s) Active</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
