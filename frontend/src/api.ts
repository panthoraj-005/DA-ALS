import {
  ApiError,
  type AppSettings,
  type Health,
  type PredictionResult,
  type SampleSignal,
  type SettingsUpdatePayload,
} from './types'

// In dev, Vite proxies /api to the backend. In the container the API and the
// built frontend share an origin, so the same relative base works there too.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? '/api'

const STATUS_TIMEOUT_MS = 8_000
// Florence-2 on CPU can take tens of seconds; the server's own timeout is 120s.
const PREDICT_TIMEOUT_MS = 180_000

/** Turn a backend-relative asset path ("/images/x.png") into something the browser can fetch. */
export function assetUrl(path: string | null): string | null {
  if (!path) return null
  return `${BASE}${path}`
}

function isOffline(error: unknown): boolean {
  return error instanceof TypeError || (error instanceof DOMException && error.name === 'AbortError')
}

async function request(
  path: string,
  init: RequestInit,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)

  // A caller-supplied signal (the chat's "Stop") aborts the same fetch the
  // timeout does; the caller tells the two apart by checking its own signal.
  const relay = () => controller.abort()
  signal?.addEventListener('abort', relay)

  try {
    return await fetch(`${BASE}${path}`, { ...init, signal: controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(
        `The server did not respond within ${Math.round(timeoutMs / 1000)}s. It may still be ` +
          'loading models, or the signal may be taking longer than expected.',
        0,
      )
    }
    if (isOffline(error)) {
      throw new ApiError(
        'Cannot reach the screening server. Check that the backend is running and try again.',
        0,
      )
    }
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', relay)
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T

  let detail = `The server returned ${response.status}.`
  try {
    const body = await response.json()
    detail = body.detail ?? body.error ?? detail
  } catch {
    /* not JSON — keep the status message */
  }

  if (response.status === 429) {
    detail = detail || 'Too many requests. Wait a moment before running another screening.'
  }
  if (response.status === 503) {
    detail = detail || 'The server is not ready to score signals yet.'
  }

  throw new ApiError(detail, response.status)
}

export async function fetchHealth(): Promise<Health> {
  return unwrap<Health>(await request('/health', {}, STATUS_TIMEOUT_MS))
}

export async function fetchSamples(): Promise<SampleSignal[]> {
  const data = await unwrap<{ samples: SampleSignal[] }>(
    await request('/samples', {}, STATUS_TIMEOUT_MS),
  )
  return data.samples
}

export interface PredictOptions {
  file?: File
  sample?: string
  explain: boolean
  useVlm: boolean
  useLlm?: boolean
}

export async function predict(options: PredictOptions): Promise<PredictionResult> {
  const form = new FormData()
  if (options.file) form.append('file', options.file)
  if (options.sample) form.append('sample', options.sample)
  form.append('explain', String(options.explain))
  form.append('use_vlm', String(options.useVlm))
  if (options.useLlm !== undefined) {
    form.append('use_llm', String(options.useLlm))
  }

  return unwrap<PredictionResult>(
    await request('/predict', { method: 'POST', body: form }, PREDICT_TIMEOUT_MS),
  )
}

export function reportUrl(id: string): string {
  return `${BASE}/report/${id}`
}

export async function sendChatMessage(
  message: string,
  history: { role: string; content: string }[],
  context?: Record<string, unknown>,
  signal?: AbortSignal,
  provider?: string,
  model?: string,
): Promise<{ reply: string; suggestions?: string[] }> {
  return await unwrap<{ reply: string; suggestions?: string[] }>(
    await request(
      '/chat',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history, context, provider, model }),
      },
      45_000,
      signal,
    ),
  )
}

export async function fetchSettings(): Promise<AppSettings> {
  return await unwrap<AppSettings>(await request('/settings', {}, STATUS_TIMEOUT_MS))
}

export async function updateSettings(payload: SettingsUpdatePayload): Promise<AppSettings> {
  return await unwrap<AppSettings>(
    await request(
      '/settings',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      STATUS_TIMEOUT_MS,
    ),
  )
}

export interface TestKeyOptions {
  provider?: string
  apiKey?: string
  model?: string
  baseUrl?: string
}

export async function testApiKey(
  apiKeyOrOptions: string | TestKeyOptions,
  legacyModel?: string,
): Promise<{ valid: boolean; message: string }> {
  let payload: Record<string, unknown>

  if (typeof apiKeyOrOptions === 'string') {
    payload = {
      provider: 'gemini',
      api_key: apiKeyOrOptions,
      gemini_api_key: apiKeyOrOptions,
      model: legacyModel,
    }
  } else {
    payload = {
      provider: apiKeyOrOptions.provider || 'gemini',
      api_key: apiKeyOrOptions.apiKey,
      gemini_api_key: apiKeyOrOptions.apiKey,
      model: apiKeyOrOptions.model,
      base_url: apiKeyOrOptions.baseUrl,
    }
  }

  return await unwrap<{ valid: boolean; message: string }>(
    await request(
      '/settings/test-key',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      },
      20_000,
    ),
  )
}

