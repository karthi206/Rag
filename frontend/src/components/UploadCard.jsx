/**
 * UploadCard.jsx — Drag-and-drop PDF file uploader
 */
import React, { useRef, useState, useCallback } from 'react'
import { Upload, FileText, CheckCircle2, AlertCircle, Loader2, X } from 'lucide-react'

export default function UploadCard({ uploading, uploadProgress, uploadError, lastResult, onUpload }) {
  const inputRef    = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [pending, setPending]   = useState([])

  const handleFiles = useCallback((files) => {
    const pdfs = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'))
    if (pdfs.length === 0) return
    setPending(pdfs)
    onUpload(pdfs)
  }, [onUpload])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }, [handleFiles])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true)  }
  const onDragLeave = ()  => setDragging(false)

  const onInputChange = (e) => handleFiles(e.target.files)

  const statusIcon = () => {
    if (uploadProgress === 'uploading' || uploadProgress === 'processing')
      return <Loader2 size={18} className="animate-spin text-violet-400" />
    if (uploadProgress === 'done')
      return <CheckCircle2 size={18} className="text-emerald-400" />
    if (uploadProgress === 'error')
      return <AlertCircle size={18} className="text-rose-400" />
    return null
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        className={`drop-zone text-center group ${dragging ? 'dragging' : ''} ${uploading ? 'pointer-events-none opacity-60' : ''}`}
        onClick={() => !uploading && inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Upload PDF documents"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={onInputChange}
          id="pdf-upload-input"
        />

        <div className="flex flex-col items-center gap-2">
          {uploading ? (
            <Loader2 size={28} className="text-violet-400 animate-spin" />
          ) : (
            <Upload
              size={28}
              className="text-slate-500 group-hover:text-violet-400 transition-colors duration-200"
            />
          )}

          <div>
            <p className="text-sm text-slate-400 group-hover:text-slate-300 transition-colors">
              {uploading
                ? uploadProgress === 'uploading' ? 'Uploading…' : 'Processing & embedding…'
                : 'Drop PDFs here or click to browse'}
            </p>
            <p className="text-xs text-slate-600 mt-0.5">PDF files only</p>
          </div>
        </div>
      </div>

      {/* Status / result */}
      {(uploadProgress || uploadError) && (
        <div className={`flex items-start gap-2 px-3 py-2 rounded-xl text-xs animate-fade-in
          ${uploadProgress === 'done' ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' : ''}
          ${uploadProgress === 'error' ? 'bg-rose-500/10 text-rose-300 border border-rose-500/20' : ''}
          ${(uploadProgress === 'uploading' || uploadProgress === 'processing') ? 'bg-violet-500/10 text-violet-300 border border-violet-500/20' : ''}
        `}>
          <div className="mt-0.5">{statusIcon()}</div>
          <div>
            {uploadProgress === 'done' && lastResult && (
              <span>
                {lastResult.loaded?.length > 0 && `✓ ${lastResult.loaded.join(', ')} ingested`}
                {lastResult.chunks > 0 && ` — ${lastResult.chunks} chunks`}
                {lastResult.skipped?.length > 0 && ` · ${lastResult.skipped.length} already existed`}
                {lastResult.failed?.length > 0 && ` · ${lastResult.failed.length} failed`}
              </span>
            )}
            {uploadProgress === 'error' && (uploadError || 'Upload failed')}
            {uploadProgress === 'uploading' && 'Sending files to server…'}
            {uploadProgress === 'processing' && 'Embedding chunks into vector store…'}
          </div>
        </div>
      )}
    </div>
  )
}
