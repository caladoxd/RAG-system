import { useState } from 'react'
import { apiUrl } from '../config'
import { readApiError } from '../api/http'

type Mode = 'file' | 'text'

export default function DocumentUpload() {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<Mode>('file')
  const [docId, setDocId] = useState('')
  const [documentText, setDocumentText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function submitFile(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setResult(null)
    const id = docId.trim()
    if (!id) {
      setError('Document id is required.')
      return
    }
    if (!file) {
      setError('Choose a file to upload.')
      return
    }
    setLoading(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      const params = new URLSearchParams({ doc_id: id })
      const res = await fetch(apiUrl(`/llm/index-file?${params}`), {
        method: 'POST',
        body: fd,
      })
      if (!res.ok) {
        setError(await readApiError(res))
        return
      }
      setResult((await res.json()) as Record<string, unknown>)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  async function submitText(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setResult(null)
    const id = docId.trim()
    const text = documentText.trim()
    if (!id) {
      setError('Document id is required.')
      return
    }
    if (!text) {
      setError('Paste or type document text.')
      return
    }
    setLoading(true)
    try {
      const res = await fetch(apiUrl('/llm/index'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_id: id,
          document: text,
        }),
      })
      if (!res.ok) {
        setError(await readApiError(res))
        return
      }
      setResult((await res.json()) as Record<string, unknown>)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="rag-panel text-left">
      <button
        type="button"
        className="rag-collapse-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="rag-panel__title">Store documents</span>
        <span className="rag-collapse-icon">{open ? '−' : '+'}</span>
      </button>

      {open && (
        <>
          <p className="rag-panel__hint mb-4">
            Upload a file (PDF, DOCX, etc.) or paste plain text. Chunks replace any prior index for
            the same <code>doc id</code>.
          </p>

          <div className="grid gap-3 max-w-xl">
            <label className="rag-field">
              <span className="rag-field__label">Document id</span>
              <input
                className="rag-input"
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
                placeholder="e.g. handbook-2024"
                autoComplete="off"
              />
            </label>
          </div>

          <div className="rag-tabs mt-4 mb-4" role="tablist" aria-label="Index mode">
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'file'}
              aria-controls="index-tabpanel-file"
              id="index-tab-file"
              className={`rag-tab ${mode === 'file' ? 'rag-tab--active' : ''}`}
              onClick={() => setMode('file')}
            >
              Upload file
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mode === 'text'}
              aria-controls="index-tabpanel-text"
              id="index-tab-text"
              className={`rag-tab ${mode === 'text' ? 'rag-tab--active' : ''}`}
              onClick={() => setMode('text')}
            >
              Paste text
            </button>
          </div>

          {mode === 'file' ? (
            <div
              role="tabpanel"
              id="index-tabpanel-file"
              aria-labelledby="index-tab-file"
              className="rag-tabpanel"
            >
              <form className="mt-4 flex flex-col gap-4 max-w-xl" onSubmit={submitFile}>
                <label className="rag-field">
                  <span className="rag-field__label">File</span>
                  <input
                    type="file"
                    className="rag-file"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                </label>
                <button type="submit" className="rag-button" disabled={loading}>
                  {loading ? 'Indexing…' : 'Index file'}
                </button>
              </form>
            </div>
          ) : (
            <div
              role="tabpanel"
              id="index-tabpanel-text"
              aria-labelledby="index-tab-text"
              className="rag-tabpanel"
            >
              <form className="mt-4 flex flex-col gap-4 max-w-xl" onSubmit={submitText}>
                <label className="rag-field">
                  <span className="rag-field__label">Plain text</span>
                  <textarea
                    className="rag-textarea"
                    value={documentText}
                    onChange={(e) => setDocumentText(e.target.value)}
                    rows={10}
                    placeholder="Paste document content…"
                  />
                </label>
                <button type="submit" className="rag-button" disabled={loading}>
                  {loading ? 'Indexing…' : 'Index text'}
                </button>
              </form>
            </div>
          )}
        </>
      )}

      {error && (
        <p className="rag-error mt-4" role="alert">
          {error}
        </p>
      )}
      {result && (
        <pre className="rag-pre mt-4 text-left">{JSON.stringify(result, null, 2)}</pre>
      )}
    </section>
  )
}
