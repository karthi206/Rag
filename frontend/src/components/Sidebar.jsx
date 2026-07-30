/**
 * Sidebar.jsx — Left panel: branding, system status, document manager, controls
 */
import React, { useEffect } from 'react'
import {
  BookOpen, Cpu, Settings2, Trash2, RefreshCw,
  FileText, ChevronRight, BarChart3
} from 'lucide-react'
import StatusBadge from './StatusBadge'
import UploadCard from './UploadCard'

function SidebarSection({ title, icon: Icon, children }) {
  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-lg bg-violet-600/20 flex items-center justify-center">
          <Icon size={12} className="text-violet-400" />
        </div>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
      </div>
      {children}
    </div>
  )
}

export default function Sidebar({
  status,
  statusLoading,
  statusError,
  documents,
  uploading,
  uploadProgress,
  uploadError,
  lastResult,
  onUpload,
  onClearDocuments,
  onClearChat,
  onRefreshStatus,
}) {
  return (
    <aside className="w-72 flex-shrink-0 h-full flex flex-col gap-4 overflow-y-auto p-4
      border-r border-white/[0.05]"
    >
      {/* ── Branding ──────────────────────────────── */}
      <div className="px-1 pt-2 pb-4 border-b border-white/[0.05]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-violet-800
            flex items-center justify-center violet-glow flex-shrink-0">
            <BookOpen size={16} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-100 leading-tight">RAG Assistant</h1>
            <p className="text-xs text-slate-500">Document Intelligence</p>
          </div>
        </div>
        <p className="mt-3 text-xs text-slate-600 leading-relaxed">
          Hybrid BM25 + vector search · Cross-encoder reranking · Ollama LLM
        </p>
      </div>

      {/* ── System Status ─────────────────────────── */}
      <SidebarSection title="System Status" icon={Cpu}>
        <StatusBadge status={status} loading={statusLoading} error={statusError} />
        {status && (
          <div className="pt-1 space-y-1.5 border-t border-white/[0.05]">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Embeddings</span>
              <span className="text-slate-400 font-mono truncate max-w-[130px]">{status.embed_model}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Top-K</span>
              <span className="text-slate-400 font-mono">{status.k_retrieve}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Chunk size</span>
              <span className="text-slate-400 font-mono">{status.chunk_size} / {status.chunk_overlap}</span>
            </div>
          </div>
        )}
        <button
          onClick={onRefreshStatus}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-violet-400 transition-colors duration-150 group"
          id="refresh-status-btn"
        >
          <RefreshCw size={11} className="group-hover:rotate-180 transition-transform duration-300" />
          Refresh status
        </button>
      </SidebarSection>

      {/* ── Document Upload ────────────────────────── */}
      <SidebarSection title="Upload Documents" icon={FileText}>
        <UploadCard
          uploading={uploading}
          uploadProgress={uploadProgress}
          uploadError={uploadError}
          lastResult={lastResult}
          onUpload={onUpload}
        />
      </SidebarSection>

      {/* ── Ingested Documents ─────────────────────── */}
      <SidebarSection title="Ingested Documents" icon={BarChart3}>
        {documents.length === 0 ? (
          <p className="text-xs text-slate-600 italic">No documents indexed yet.</p>
        ) : (
          <ul className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {documents.map((doc, idx) => (
              <li key={idx} className="flex items-center gap-2 group">
                <ChevronRight size={10} className="text-violet-500 flex-shrink-0" />
                <span className="text-xs text-slate-400 group-hover:text-slate-200
                  truncate transition-colors duration-150 font-mono">
                  {doc}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-slate-600">{documents.length} document(s)</p>
      </SidebarSection>

      {/* ── Controls ──────────────────────────────── */}
      <SidebarSection title="Controls" icon={Settings2}>
        <div className="space-y-2">
          <button
            onClick={onClearChat}
            className="btn-ghost w-full flex items-center gap-2 text-xs justify-start"
            id="clear-chat-btn"
          >
            <Trash2 size={13} />
            Clear Chat History
          </button>
          <button
            onClick={onClearDocuments}
            className="btn-ghost w-full flex items-center gap-2 text-xs justify-start text-rose-400/70 hover:text-rose-400"
            id="clear-docs-btn"
          >
            <Trash2 size={13} />
            Clear All Documents
          </button>
        </div>
      </SidebarSection>
    </aside>
  )
}
