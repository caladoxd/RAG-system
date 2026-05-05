export async function readApiError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    const d = data?.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) {
      return d
        .map((x: { msg?: string; loc?: unknown }) => x.msg ?? JSON.stringify(x))
        .join('; ')
    }
    if (d && typeof d === 'object') return JSON.stringify(d)
    return res.statusText || `HTTP ${res.status}`
  } catch {
    return res.statusText || `HTTP ${res.status}`
  }
}
