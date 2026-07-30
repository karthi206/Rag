/**
 * SourceChips.jsx — Renders attributed source document chips
 */
import React from 'react'
import { FileText } from 'lucide-react'

export default function SourceChips({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div className="flex flex-wrap gap-2 mt-3 pt-3 border-t border-white/[0.05]">
      <span className="text-xs text-slate-500 self-center">Sources:</span>
      {sources.map((src, idx) => (
        <span key={idx} className="source-chip">
          <FileText size={10} />
          {src}
        </span>
      ))}
    </div>
  )
}
