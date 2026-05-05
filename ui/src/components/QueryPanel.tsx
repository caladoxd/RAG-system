import { useState } from 'react'
import { apiUrl } from '../config'
import { readApiError } from '../api/http'

export type ContextChunk = {
  doc_id?: string
  section_id?: string
  chunk_index?: number
  text?: string
  rerank_score?: number
  vector_distance?: number
}

export type QueryResponseBody = {
  answer: string
  context_results: ContextChunk[]
  metrics?: Record<string, unknown> | null
}

export default function QueryPanel() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [docIdFilter, setDocIdFilter] = useState('')
  const [metrics, setMetrics] = useState(false)
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState<QueryResponseBody | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    const q = query.trim()
    if (!q) {
      setError('Enter a question.')
      return
    }
    setLoading(true)
    setData(null)
    try {
      const body: Record<string, unknown> = {
        query: q,
        top_k: topK,
        hybrid: true,
        cross_encoder: true,
        metrics,
      }
      const df = docIdFilter.trim()
      if (df) body.doc_id = df

      const res = await fetch(apiUrl('/llm/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        setError(await readApiError(res))
        return
      }
      setData((await res.json()) as QueryResponseBody)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="rag-panel text-left">
      <h2 className="rag-panel__title">Ask the corpus</h2>
      <p className="rag-panel__hint mb-4">
        Runs retrieval + generation against your indexed chunks. Optionally restrict to one{' '}
        <code>doc id</code>.
      </p>

      <form className="flex flex-col gap-4 max-w-xl" onSubmit={submit}>
        <label className="rag-field">
          <span className="rag-field__label">Question</span>
          <textarea
            className="rag-textarea"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            placeholder="What does the policy say about…?"
          />
        </label>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label className="rag-field">
            <span className="rag-field__label">Top-K chunks</span>
            <input
              type="number"
              className="rag-input"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </label>
          <label className="rag-field">
            <span className="rag-field__label">Filter by doc id (optional)</span>
            <input
              className="rag-input"
              value={docIdFilter}
              onChange={(e) => setDocIdFilter(e.target.value)}
              placeholder="Any indexed doc"
              autoComplete="off"
            />
          </label>
        </div>

        <label className="rag-check">
          <input
            type="checkbox"
            checked={metrics}
            onChange={(e) => setMetrics(e.target.checked)}
          />
          <span>Include diagnostics (recall, citations, faithfulness—slower)</span>
        </label>

        <button type="submit" className="rag-button self-start" disabled={loading}>
          {loading ? 'Running…' : 'Run query'}
        </button>
      </form>

      {error && (
        <p className="rag-error mt-4" role="alert">
          {error}
        </p>
      )}

      {data && (
        <div className="mt-6 space-y-4">
          <div>
            <h3 className="text-lg font-medium text-[var(--text-h)] mb-2">Answer</h3>
            <div className="rag-answer whitespace-pre-wrap">{data.answer}</div>
          </div>

          {data.metrics != null && (
            <details className="rag-details">
              <summary className="cursor-pointer text-[var(--text-h)]">Metrics</summary>
              <pre className="rag-pre mt-2">{JSON.stringify(data.metrics, null, 2)}</pre>
            </details>
          )}
        </div>
      )}
    </section>
  )
}
