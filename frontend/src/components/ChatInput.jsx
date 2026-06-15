/**
 * ChatInput.jsx — Bottom chat input bar
 */
import React, { useState, useRef, useCallback } from 'react'
import { Send, Square, Loader2 } from 'lucide-react'

export default function ChatInput({ onSend, onStop, streaming, disabled, noDocuments }) {
  const [value, setValue]   = useState('')
  const textareaRef         = useRef(null)

  const handleSubmit = useCallback((e) => {
    e?.preventDefault()
    if (!value.trim() || streaming || disabled) return
    onSend(value.trim())
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [value, streaming, disabled, onSend])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleTextareaChange = (e) => {
    setValue(e.target.value)
    // Auto-resize
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }
  }

  const placeholder = noDocuments
    ? 'Upload PDF documents first to start chatting…'
    : streaming
      ? 'Generating answer…'
      : 'Ask a question about your documents… (Enter to send, Shift+Enter for newline)'

  return (
    <div className="px-4 py-4 border-t border-white/[0.05]">
      <form
        onSubmit={handleSubmit}
        className="glass-card p-2 flex items-end gap-2 focus-within:border-violet-500/40 transition-all duration-200"
      >
        <textarea
          ref={textareaRef}
          id="chat-input"
          value={value}
          onChange={handleTextareaChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || noDocuments}
          rows={1}
          className="flex-1 bg-transparent text-sm text-slate-100 placeholder:text-slate-600
            resize-none focus:outline-none leading-relaxed py-2 px-2 max-h-40 overflow-y-auto
            disabled:opacity-40 disabled:cursor-not-allowed"
          style={{ minHeight: '40px' }}
          aria-label="Chat input"
        />

        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            id="stop-streaming-btn"
            className="flex-shrink-0 w-9 h-9 rounded-xl bg-rose-500/20 border border-rose-500/30
              text-rose-400 hover:bg-rose-500/30 transition-all duration-200
              flex items-center justify-center"
            title="Stop generation"
          >
            <Square size={14} />
          </button>
        ) : (
          <button
            type="submit"
            id="send-message-btn"
            disabled={!value.trim() || disabled || noDocuments}
            className="btn-primary flex-shrink-0 w-9 h-9 p-0 flex items-center justify-center"
            title="Send message"
          >
            <Send size={14} />
          </button>
        )}
      </form>

      <p className="text-center text-xs text-slate-700 mt-2">
        Answers are grounded exclusively in your uploaded documents.
      </p>
    </div>
  )
}
