/**
 * useStatus.js — Polls /api/status every 15 seconds for system health info.
 */
import { useState, useEffect, useCallback } from 'react'

export default function useStatus() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState(null)

  const fetchStatus = useCallback(async () => {
    try {
      const res  = await fetch('/api/status')
      const data = await res.json()
      setStatus(data)
      setError(null)
    } catch (err) {
      setError('Cannot reach backend')
      setStatus(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 15_000)
    return () => clearInterval(id)
  }, [fetchStatus])

  return { status, loading, error, refresh: fetchStatus }
}
