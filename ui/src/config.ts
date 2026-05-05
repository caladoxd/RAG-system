/** Base URL for the FastAPI app. In dev, default uses Vite proxy (`/api` → backend). */
export function apiUrl(path: string): string {
  const p = path.startsWith('/') ? path : `/${path}`
  const explicit = import.meta.env.VITE_API_URL?.replace(/\/$/, '')
  if (explicit) return `${explicit}${p}`
  if (import.meta.env.DEV) return `/api${p}`
  return `http://127.0.0.1:8000${p}`
}
