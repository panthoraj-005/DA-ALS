export interface ShapFeature {
  feature: string
  label: string
  value: number
  direction: string
}

export interface PredictionResult {
  id: string
  source: string
  final_prediction: 'ALS' | 'Normal'
  final_probability: number
  final_confidence: number
  severity: 'Severe' | 'Moderate' | 'Mild' | 'N/A'
  cnn_probability: number
  cnn_prediction: 'ALS' | 'Normal'
  florence_probability: number | null
  florence_prediction: 'ALS' | 'Normal' | null
  florence_confidence: number | null
  florence_caption: string | null
  meta_probability: number | null
  models_agree: boolean | null
  fusion: 'cnn_only' | 'cnn+florence' | 'cnn+florence+meta' | string
  fusion_note: string | null
  abnormal_segments: string
  abnormal_segment_count: number
  n_segments: number
  segment_energies: number[]
  signal_length: number
  signal_image_url: string
  gradcam_image_url: string | null
  gradcam_available?: boolean
  gradcam_profile?: number[]
  gradcam_peak_percent?: number
  top_shap_feature: string | null
  top_shap_features: ShapFeature[] | null
  explanation: string
  explanation_source?: string
  demo_mode: boolean
  device: string
  disclaimer: string
  timings: Record<string, number>
}

export interface Artifact {
  name: string
  loaded: boolean
  path: string | null
  detail: string
}

export interface Health {
  status: 'ready' | 'degraded' | 'demo' | 'unavailable'
  device: string
  demo_mode: boolean
  capabilities: {
    cnn: boolean
    florence: boolean
    meta_learner: boolean
    full_fusion: boolean
    gradcam: boolean
    shap: boolean
    llm_explanation: boolean
  }
  artifacts: Artifact[]
  config: {
    signal_length: number
    fs: number
    bandpass: [number, number]
    n_segments: number
    meta_feature_dim: number
    florence_default: boolean
    max_upload_mb: number
    max_batch_size: number
  }
  disclaimer: string
}

export interface SampleSignal {
  name: string
  label: string
  size_kb: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  suggestions?: string[]
  /** Set on an assistant message that reports a failed request, so it can offer a retry. */
  failedQuery?: string
}

export interface ProviderModel {
  id: string
  name: string
}

export interface ProviderInfo {
  id: string
  name: string
  description: string
  env_key: string
  requires_key: boolean
  key_url: string
  configured: boolean
  key_masked: string
  default_model: string
  models: ProviderModel[]
}

export interface AppSettings {
  llm_provider: string
  llm_model: string
  openai_base_url?: string
  ollama_base_url?: string
  gemini_api_key_masked: string
  gemini_api_key_configured: boolean
  enable_llm: boolean
  enable_florence: boolean
  florence_default: boolean
  available_models: { id: string; name: string }[]
  available_providers: { id: string; name: string }[]
  providers?: ProviderInfo[]
}

export interface SettingsUpdatePayload {
  gemini_api_key?: string
  openai_api_key?: string
  openai_base_url?: string
  anthropic_api_key?: string
  groq_api_key?: string
  openrouter_api_key?: string
  ollama_base_url?: string
  llm_model?: string
  llm_provider?: string
  enable_llm?: boolean
  florence_default?: boolean
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

