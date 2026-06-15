/**
 * useChat.js — Manages chat history and streams responses from /api/chat.
 *
 * The backend streams plain text tokens terminated by an optional
 * "|||SOURCES|||source1|||source2" marker at the very end.
 */
import { useState, useCallback, useRef } from 'react'

const SOURCES_MARKER = '|||SOURCES|||'

export default function useChat() {
  const [messages,  setMessages]  = useState([])  // { role, content, sources? }
  const [streaming, setStreaming]  = useState(false)
  const [streamText, setStreamText] = useState('')
  const [error, setError]          = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || streaming) return

    // Add user message immediately
    const userMsg = { role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setStreaming(true)
    setStreamText('')
    setError(null)

    // Build history for the request (exclude the message we just appended)
    const historyForRequest = messages.map(m => [m.role, m.content])

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history: historyForRequest }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        accumulated += chunk
        setStreamText(accumulated)
      }

      // Extract sources from the end of the stream
      let content = accumulated
      let sources = []
      const markerIdx = accumulated.indexOf(SOURCES_MARKER)
      if (markerIdx !== -1) {
        content = accumulated.slice(0, markerIdx)
        sources = accumulated.slice(markerIdx + SOURCES_MARKER.length).split('|||').filter(Boolean)
      }

      const assistantMsg = { role: 'assistant', content: content.trim(), sources }
      setMessages(prev => [...prev, assistantMsg])

    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
        const errMsg = {
          role: 'assistant',
          content: `⚠️ **Error:** ${err.message}`,
          sources: [],
        }
        setMessages(prev => [...prev, errMsg])
      }
    } finally {
      setStreaming(false)
      setStreamText('')
      abortRef.current = null
    }
  }, [messages, streaming])

  const stopStreaming = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
    }
  }, [])

  const clearHistory = useCallback(() => {
    setMessages([])
    setError(null)
  }, [])

  return { messages, streaming, streamText, error, sendMessage, stopStreaming, clearHistory }
}
