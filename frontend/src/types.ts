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
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}
