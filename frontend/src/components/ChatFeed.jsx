/**
 * ChatFeed.jsx — Displays the conversation history + live streaming bubble
 */
import React, { useEffect, useRef } from 'react'
import { Bot, User, AlertTriangle } from 'lucide-react'
import SourceChips from './SourceChips'

function MessageBubble({ message }) {
  const isUser      = message.role === 'user'
  const isError     = message.content?.startsWith('⚠️')

  return (
    <div className={`flex gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
        ${isUser
          ? 'bg-violet-600/30 border border-violet-500/30'
          : 'bg-slate-700/50 border border-white/[0.06]'
        }`}
      >
        {isUser
          ? <User  size={14} className="text-violet-300" />
          : <Bot   size={14} className="text-slate-300" />
        }
      </div>

      {/* Bubble */}
      <div className={`max-w-[80%] space-y-1 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        <div className={`
          px-4 py-3 rounded-2xl text-sm leading-relaxed
          ${isUser
            ? 'bg-violet-600/20 border border-violet-500/20 text-slate-100 rounded-tr-md'
            : isError
              ? 'bg-rose-500/10 border border-rose-500/20 text-rose-300 rounded-tl-md'
              : 'glass-card text-slate-200 rounded-tl-md'
          }
        `}>
          <p className="whitespace-pre-wrap break-words prose-rag">{message.content}</p>
        </div>

        {/* Source chips for assistant messages */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="px-1 w-full">
            <SourceChips sources={message.sources} />
          </div>
        )}
      </div>
    </div>
  )
}

function StreamingBubble({ text }) {
  return (
    <div className="flex gap-3 animate-fade-in">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
        bg-slate-700/50 border border-white/[0.06]">
        <Bot size={14} className="text-violet-400 animate-pulse" />
      </div>

      {/* Streaming bubble */}
      <div className="max-w-[80%]">
        <div className="glass-card px-4 py-3 rounded-2xl rounded-tl-md">
          {text ? (
            <p className="text-sm text-slate-200 whitespace-pre-wrap break-words leading-relaxed streaming-cursor">
              {text}
            </p>
          ) : (
            <div className="flex gap-1.5 items-center py-1">
              <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"   style={{ animationDelay: '0ms'   }} />
              <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"   style={{ animationDelay: '150ms' }} />
              <div className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-bounce"   style={{ animationDelay: '300ms' }} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function ChatFeed({ messages, streaming, streamText }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom on new messages / stream updates
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, streamText])

  const isEmpty = messages.length === 0 && !streaming

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
      {isEmpty && (
        <div className="flex flex-col items-center justify-center h-full gap-4 text-center animate-fade-in">
          <div className="w-16 h-16 rounded-2xl bg-violet-600/10 border border-violet-500/20
            flex items-center justify-center violet-glow">
            <Bot size={28} className="text-violet-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold gradient-text mb-2">
              RAG Document Assistant
            </h2>
            <p className="text-sm text-slate-500 max-w-md leading-relaxed">
              Upload your PDF documents using the sidebar, then ask questions.
              Answers are grounded in your documents using hybrid search and cross-encoder reranking.
            </p>
          </div>
        </div>
      )}

      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}

      {streaming && (
        <StreamingBubble text={streamText} />
      )}

      <div ref={bottomRef} />
    </div>
  )
}
