/**
 * StatusBadge.jsx — Shows Ollama server connectivity status
 */
import React from 'react'
import { Wifi, WifiOff, Loader2, Cpu } from 'lucide-react'

export default function StatusBadge({ status, loading, error }) {
  if (loading) {
    return (
      <div className="badge-loading">
        <Loader2 size={12} className="animate-spin" />
        <span>Connecting…</span>
      </div>
    )
  }

  if (error || !status) {
    return (
      <div className="badge-offline">
        <WifiOff size={12} />
        <span>Backend offline</span>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {/* Ollama connectivity */}
      <div className={status.ollama ? 'badge-online' : 'badge-offline'}>
        {status.ollama ? <Wifi size={12} /> : <WifiOff size={12} />}
        <span>Ollama {status.ollama ? 'connected' : 'offline'}</span>
      </div>

      {/* Active model */}
      {status.ollama && (
        <div className="flex items-center gap-1.5">
          <Cpu size={11} className="text-violet-400" />
          <span className="text-xs text-slate-400 font-mono">{status.active_model}</span>
        </div>
      )}

      {/* Chunks indexed */}
      <div className="text-xs text-slate-500">
        {status.chunks_indexed.toLocaleString()} chunks indexed
      </div>
    </div>
  )
}
