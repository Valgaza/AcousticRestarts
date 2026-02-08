'use client'

import { useState, useMemo, useRef } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown, ChevronUp, TrendingUp, AlertTriangle, Upload, CheckCircle, XCircle, Navigation, Plus, Trash2 } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useTrafficSimulation } from '@/hooks/useTrafficSimulation'
import type { useLiveTraffic } from '@/hooks/useLiveTraffic'
import type { useGridTraffic } from '@/hooks/useGridTraffic'
import PredictionChart from './PredictionChart'
import { uploadCsvFile, optimizeRoutes, type RouteRequest } from '@/lib/trafficApi'

interface AnalyticsPanelProps {
  state: ReturnType<typeof useTrafficSimulation>
  liveTraffic: ReturnType<typeof useLiveTraffic>
  gridTraffic: ReturnType<typeof useGridTraffic>
  useLiveData?: boolean
}

export default function AnalyticsPanel({ state, liveTraffic, gridTraffic, useLiveData = true }: AnalyticsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [uploadMessage, setUploadMessage] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Route optimization state
  const [routeRequests, setRouteRequests] = useState<RouteRequest[]>([
    { start_node: 0, end_node: 150, timestamp: '2026-02-08T03:00:00' }
  ])
  const [optimizeStatus, setOptimizeStatus] = useState<'idle' | 'optimizing' | 'success' | 'error'>('idle')
  const [optimizeMessage, setOptimizeMessage] = useState('')
  const [routeResults, setRouteResults] = useState<any>(null)

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

  const handleAddRouteRequest = () => {
    setRouteRequests([
      ...routeRequests,
      { start_node: 0, end_node: 150, timestamp: '2026-02-08T03:00:00' }
    ])
  }

  const handleRemoveRouteRequest = (index: number) => {
    setRouteRequests(routeRequests.filter((_, i) => i !== index))
  }

  const handleRouteRequestChange = (index: number, field: keyof RouteRequest, value: string | number) => {
    const updated = [...routeRequests]
    updated[index] = { ...updated[index], [field]: value }
    setRouteRequests(updated)
  }

  const handleOptimizeRoutes = async () => {
    setOptimizeStatus('optimizing')
    setOptimizeMessage('Optimizing routes...')
    setRouteResults(null)

    try {
      const result = await optimizeRoutes(routeRequests)
      setOptimizeStatus('success')
      setOptimizeMessage(`${result.assignments.length} route(s) optimized successfully`)
      setRouteResults(result)
      
      // Clear success message after 10 seconds
      setTimeout(() => {
        setOptimizeStatus('idle')
        setOptimizeMessage('')
      }, 10000)
    } catch (error) {
      setOptimizeStatus('error')
      setOptimizeMessage(error instanceof Error ? error.message : 'Optimization failed')
      
      // Clear error message after 5 seconds
      setTimeout(() => {
        setOptimizeStatus('idle')
        setOptimizeMessage('')
      }, 5000)
    }
  }

  // Calculate metrics from grid data
  const gridMetrics = useMemo(() => {
    const cells = gridTraffic.grid.flat()
    const congestionLevels = cells.map(c => c.congestionLevel).filter(c => c > 0)
    
    const avgCongestion = congestionLevels.length > 0 
      ? congestionLevels.reduce((a, b) => a + b, 0) / congestionLevels.length 
      : 0
    
    const avgSpeed = Math.round(60 * (1 - avgCongestion * 0.7))
    // Count red (>= 0.6) and purple (>= 0.8) areas as bottlenecks
    const bottlenecks = cells.filter(c => c.congestionLevel >= 0.6).length
    
    return {
      avgCitySpeed: avgSpeed,
      activeBottlenecks: bottlenecks
    }
  }, [gridTraffic.grid])

  // Use grid metrics
  const metrics = gridMetrics

  const stats = [
    {
      label: 'Avg City Speed',
      value: `${metrics.avgCitySpeed} km/h`,
      icon: TrendingUp,
      color: 'text-emerald-400',
      change: `t+${gridTraffic.currentFrameIndex}h forecast`,
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

            {/* Route Optimization Section */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Route Optimization
              </p>
              <Card className="bg-slate-800 border-slate-700 p-4">
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-slate-300">
                    <Navigation className="w-4 h-4" />
                    <span className="text-xs font-mono">Find Optimal Routes</span>
                  </div>

                  {/* Route Requests */}
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {routeRequests.map((request, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="p-3 bg-slate-900 border border-slate-700 space-y-2"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-mono text-slate-400">Route {index + 1}</span>
                          {routeRequests.length > 1 && (
                            <Button
                              onClick={() => handleRemoveRouteRequest(index)}
                              variant="ghost"
                              size="sm"
                              className="h-5 w-5 p-0 text-rose-400 hover:text-rose-300 hover:bg-rose-500/10"
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          )}
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <label className="text-xs font-mono text-slate-500 block mb-1">
                              Start Node
                            </label>
                            <Input
                              type="number"
                              value={request.start_node}
                              onChange={(e) => handleRouteRequestChange(index, 'start_node', parseInt(e.target.value) || 0)}
                              className="h-8 bg-slate-800 border-slate-600 text-slate-100 text-xs font-mono"
                              placeholder="0"
                            />
                          </div>
                          <div>
                            <label className="text-xs font-mono text-slate-500 block mb-1">
                              End Node
                            </label>
                            <Input
                              type="number"
                              value={request.end_node}
                              onChange={(e) => handleRouteRequestChange(index, 'end_node', parseInt(e.target.value) || 0)}
                              className="h-8 bg-slate-800 border-slate-600 text-slate-100 text-xs font-mono"
                              placeholder="150"
                            />
                          </div>
                        </div>
                        
                        <div>
                          <label className="text-xs font-mono text-slate-500 block mb-1">
                            Timestamp
                          </label>
                          <Input
                            type="text"
                            value={request.timestamp}
                            onChange={(e) => handleRouteRequestChange(index, 'timestamp', e.target.value)}
                            className="h-8 bg-slate-800 border-slate-600 text-slate-100 text-xs font-mono"
                            placeholder="2026-02-08T03:00:00"
                          />
                        </div>
                      </motion.div>
                    ))}
                  </div>

                  {/* Add Route Button */}
                  <Button
                    onClick={handleAddRouteRequest}
                    variant="outline"
                    size="sm"
                    className="w-full border-slate-600 text-slate-300 hover:bg-slate-700 hover:text-slate-100"
                  >
                    <Plus className="w-3 h-3 mr-2" />
                    Add Route
                  </Button>

                  {/* Optimize Button */}
                  <Button
                    onClick={handleOptimizeRoutes}
                    disabled={optimizeStatus === 'optimizing' || routeRequests.length === 0}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                    size="sm"
                  >
                    {optimizeStatus === 'optimizing' ? (
                      <>
                        <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                        Optimizing...
                      </>
                    ) : (
                      <>
                        <Navigation className="w-3 h-3 mr-2" />
                        Optimize Routes
                      </>
                    )}
                  </Button>

                  {/* Status Message */}
                  {optimizeMessage && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`flex items-start gap-2 p-2 border text-xs font-mono ${
                        optimizeStatus === 'success'
                          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                          : optimizeStatus === 'error'
                          ? 'bg-rose-500/10 border-rose-500/30 text-rose-400'
                          : 'bg-slate-700 border-slate-600 text-slate-300'
                      }`}
                    >
                      {optimizeStatus === 'success' && <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />}
                      {optimizeStatus === 'error' && <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />}
                      <span className="flex-1">{optimizeMessage}</span>
                    </motion.div>
                  )}

                  {/* Results Display */}
                  {routeResults && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-2"
                    >
                      {routeResults.assignments.map((assignment: any, idx: number) => (
                        <div key={idx} className="p-2 bg-slate-900 border border-emerald-500/30 text-xs font-mono">
                          <div className="text-emerald-400 mb-1">Route {idx + 1} Results:</div>
                          <div className="text-slate-300 space-y-1">
                            <div>Nodes: {assignment.route_nodes.length} nodes</div>
                            <div>Cost: {assignment.total_cost.toFixed(2)}</div>
                            <div>Alternatives: {assignment.alternatives_considered}</div>
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  )}

                  <p className="text-xs text-slate-500 font-mono">
                    Enter start/end nodes and timestamp
                  </p>
                </div>
              </Card>
            </div>

            {/* Prediction Chart */}
            <div>
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
                Traffic Congestion Forecast
              </p>
              <Card className="bg-slate-800 border-slate-700 p-4">
                <PredictionChart gridTraffic={gridTraffic} />
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
