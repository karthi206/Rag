/**
 * useDocuments.js — Manages document upload and listing.
 */
import { useState, useCallback } from 'react'

export default function useDocuments(onUploadComplete) {
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null) // null | 'uploading' | 'processing' | 'done' | 'error'
  const [uploadError, setUploadError] = useState(null)
  const [lastResult, setLastResult] = useState(null)

  const fetchDocuments = useCallback(async () => {
    try {
      const res  = await fetch('/api/documents')
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch {
      // Silently fail on list fetch
    }
  }, [])

  const uploadFiles = useCallback(async (files) => {
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadProgress('uploading')
    setUploadError(null)

    const formData = new FormData()
    for (const file of files) {
      formData.append('files', file)
    }

    try {
      setUploadProgress('processing')
      const res  = await fetch('/api/upload', { method: 'POST', body: formData })
      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.detail || 'Upload failed')
      }

      setLastResult(data)
      setUploadProgress('done')
      await fetchDocuments()
      if (onUploadComplete) onUploadComplete(data)

    } catch (err) {
      setUploadError(err.message)
      setUploadProgress('error')
    } finally {
      setUploading(false)
    }
  }, [fetchDocuments, onUploadComplete])

  const clearDocuments = useCallback(async () => {
    try {
      await fetch('/api/documents', { method: 'DELETE' })
      setDocuments([])
      setLastResult(null)
      setUploadProgress(null)
    } catch (err) {
      console.error('Clear failed', err)
    }
  }, [])

  return {
    documents,
    uploading,
    uploadProgress,
    uploadError,
    lastResult,
    uploadFiles,
    fetchDocuments,
    clearDocuments,
  }
}
