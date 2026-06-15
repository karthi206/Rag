/**
 * App.jsx — Root application component
 * Layout: Left Sidebar | Right Chat Panel (feed + input)
 */
import React, { useEffect } from 'react'

import Sidebar    from './components/Sidebar'
import ChatFeed   from './components/ChatFeed'
import ChatInput  from './components/ChatInput'

import useStatus    from './hooks/useStatus'
import useDocuments from './hooks/useDocuments'
import useChat      from './hooks/useChat'

export default function App() {
  // ── Status polling ──────────────────────────────────────────
  const { status, loading: statusLoading, error: statusError, refresh: refreshStatus } = useStatus()

  // ── Document management ─────────────────────────────────────
  const {
    documents,
    uploading,
    uploadProgress,
    uploadError,
    lastResult,
    uploadFiles,
    fetchDocuments,
    clearDocuments,
  } = useDocuments(() => refreshStatus())

  // ── Chat ─────────────────────────────────────────────────────
  const {
    messages,
    streaming,
    streamText,
    sendMessage,
    stopStreaming,
    clearHistory,
  } = useChat()

  // Fetch document list on mount
  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])

  const noDocuments = documents.length === 0 && !statusLoading

  const handleClearDocuments = async () => {
    await clearDocuments()
    clearHistory()
    refreshStatus()
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-animated-gradient">
      {/* ── Left sidebar ──────────────────────────────────────── */}
      <Sidebar
        status={status}
        statusLoading={statusLoading}
        statusError={statusError}
        documents={documents}
        uploading={uploading}
        uploadProgress={uploadProgress}
        uploadError={uploadError}
        lastResult={lastResult}
        onUpload={uploadFiles}
        onClearDocuments={handleClearDocuments}
        onClearChat={clearHistory}
        onRefreshStatus={refreshStatus}
      />

      {/* ── Main chat panel ───────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 h-full">
        {/* Header bar */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-white/[0.05] flex-shrink-0">
          <div>
            <h1 className="text-base font-semibold text-slate-100">
              Document Chat
            </h1>
            <p className="text-xs text-slate-500 mt-0.5">
              {messages.length > 0
                ? `${messages.length} message${messages.length > 1 ? 's' : ''} · Hybrid search + cross-encoder reranking`
                : 'Hybrid BM25 + vector search · Cross-encoder reranking · Streaming LLM'
              }
            </p>
          </div>

          {/* Status pill */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium
            ${status?.ollama
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${status?.ollama ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
            {statusLoading ? 'Connecting…' : status?.ollama ? `${status.active_model} ready` : 'Ollama offline'}
          </div>
        </header>

        {/* Warning banner if no docs */}
        {noDocuments && (
          <div className="mx-4 mt-4 px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/20
            text-amber-300 text-sm flex items-center gap-2 animate-fade-in flex-shrink-0">
            <span className="text-lg">📄</span>
            <span>Upload PDF documents in the sidebar to start asking questions.</span>
          </div>
        )}

        {/* Chat feed (scrollable) */}
        <ChatFeed
          messages={messages}
          streaming={streaming}
          streamText={streamText}
        />

        {/* Chat input (fixed at bottom) */}
        <ChatInput
          onSend={sendMessage}
          onStop={stopStreaming}
          streaming={streaming}
          disabled={!status?.ollama}
          noDocuments={noDocuments}
        />
      </main>
    </div>
  )
}
