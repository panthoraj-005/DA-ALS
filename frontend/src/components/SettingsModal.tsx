import { useEffect, useMemo, useState } from 'react'
import { fetchSettings, testApiKey, updateSettings } from '../api'
import type { AppSettings, ProviderInfo, SettingsUpdatePayload } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  onSettingsSaved?: () => void
}

const FALLBACK_PROVIDERS: ProviderInfo[] = [
  {
    id: 'gemini',
    name: 'Google Gemini',
    description: 'Google AI Studio API (Fast, generous free tier, clinical-grade reasoning)',
    env_key: 'GEMINI_API_KEY',
    requires_key: true,
    key_url: 'https://aistudio.google.com/app/apikey',
    configured: false,
    key_masked: '',
    default_model: 'gemini-3.1-flash-lite',
    models: [
      { id: 'gemini-3.1-flash-lite', name: 'Gemini 3.1 Flash Lite (Fastest, Recommended)' },
      { id: 'gemini-2.5-flash', name: 'Gemini 2.5 Flash (Balanced Clinical Reasoning)' },
      { id: 'gemini-3.7-flash', name: 'Gemini 3.7 Flash (Deep Reasoning)' },
      { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
      { id: 'gemini-1.5-pro', name: 'Gemini 1.5 Pro' },
    ],
  },
  {
    id: 'openai',
    name: 'OpenAI',
    description: 'OpenAI official API (GPT-4o, GPT-4o-mini)',
    env_key: 'OPENAI_API_KEY',
    requires_key: true,
    key_url: 'https://platform.openai.com/api-keys',
    configured: false,
    key_masked: '',
    default_model: 'gpt-4o-mini',
    models: [
      { id: 'gpt-4o-mini', name: 'GPT-4o Mini (Fast & Cost-Effective, Recommended)' },
      { id: 'gpt-4o', name: 'GPT-4o (Omni Clinical Intelligence)' },
      { id: 'o3-mini', name: 'o3-mini (Advanced Reasoning)' },
      { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
    ],
  },
  {
    id: 'anthropic',
    name: 'Anthropic Claude',
    description: 'Anthropic Claude API (Safe clinical summarization)',
    env_key: 'ANTHROPIC_API_KEY',
    requires_key: true,
    key_url: 'https://console.anthropic.com/settings/keys',
    configured: false,
    key_masked: '',
    default_model: 'claude-3-5-haiku-20241022',
    models: [
      { id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku (High Speed, Concise)' },
      { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet (State-of-the-Art Analysis)' },
      { id: 'claude-3-7-sonnet-20250219', name: 'Claude 3.7 Sonnet (Hybrid Reasoning)' },
    ],
  },
  {
    id: 'groq',
    name: 'Groq Cloud (LPU Inference)',
    description: 'Ultra high-speed open-source model inference on LPUs',
    env_key: 'GROQ_API_KEY',
    requires_key: true,
    key_url: 'https://console.groq.com/keys',
    configured: false,
    key_masked: '',
    default_model: 'llama-3.3-70b-versatile',
    models: [
      { id: 'llama-3.3-70b-versatile', name: 'Llama 3.3 70B Versatile (Recommended)' },
      { id: 'llama-3.1-8b-instant', name: 'Llama 3.1 8B Instant (Sub-second response)' },
      { id: 'deepseek-r1-distill-llama-70b', name: 'DeepSeek R1 Distill Llama 70B' },
      { id: 'mixtral-8x7b-32768', name: 'Mixtral 8x7B' },
    ],
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    description: 'Unified API gateway supporting leading open and proprietary models',
    env_key: 'OPENROUTER_API_KEY',
    requires_key: true,
    key_url: 'https://openrouter.ai/keys',
    configured: false,
    key_masked: '',
    default_model: 'google/gemini-2.5-flash',
    models: [
      { id: 'google/gemini-2.5-flash', name: 'Gemini 2.5 Flash via OpenRouter' },
      { id: 'anthropic/claude-3.5-haiku', name: 'Claude 3.5 Haiku via OpenRouter' },
      { id: 'meta-llama/llama-3.3-70b-instruct', name: 'Llama 3.3 70B Instruct' },
      { id: 'deepseek/deepseek-chat', name: 'DeepSeek V3 Chat' },
    ],
  },
  {
    id: 'ollama',
    name: 'Ollama (Local / Self-Hosted)',
    description: 'Run open models completely private & offline on your own machine/server',
    env_key: 'OLLAMA_BASE_URL',
    requires_key: false,
    key_url: 'https://ollama.com',
    configured: true,
    key_masked: 'http://localhost:11434',
    default_model: 'medgemma',
    models: [
      { id: 'medgemma', name: 'MedGemma (Medical Fine-tune)' },
      { id: 'llama3.2', name: 'Llama 3.2 (Lightweight Local)' },
      { id: 'llama3.3', name: 'Llama 3.3 (High Capability)' },
      { id: 'mistral', name: 'Mistral 7B' },
      { id: 'qwen2.5', name: 'Qwen 2.5' },
    ],
  },
  {
    id: 'local',
    name: 'Local HuggingFace Model',
    description: 'Built-in Qwen2.5-1.5B loaded directly into Python runtime',
    env_key: '',
    requires_key: false,
    key_url: '',
    configured: true,
    key_masked: 'Loaded in memory',
    default_model: 'Qwen/Qwen2.5-1.5B',
    models: [{ id: 'Qwen/Qwen2.5-1.5B', name: 'Qwen2.5-1.5B (Local Weights)' }],
  },
]

export default function SettingsModal({ open, onClose, onSettingsSaved }: Props) {
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [settings, setSettings] = useState<AppSettings | null>(null)

  // Active settings
  const [activeProvider, setActiveProvider] = useState('gemini')
  const [activeModel, setActiveModel] = useState('gemini-3.1-flash-lite')
  const [enableLlm, setEnableLlm] = useState(true)
  const [florenceDefault, setFlorenceDefault] = useState(true)

  // Selected tab in the configuration UI
  const [currentTab, setCurrentTab] = useState('gemini')

  // Per-provider credential inputs
  const [keysInput, setKeysInput] = useState<Record<string, string>>({
    gemini: '',
    openai: '',
    anthropic: '',
    groq: '',
    openrouter: '',
    ollama: 'http://localhost:11434',
  })
  const [openaiBaseUrl, setOpenaiBaseUrl] = useState('')
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({})

  const [testResults, setTestResults] = useState<Record<string, { valid: boolean; message: string }>>({})
  const [saveSuccess, setSaveSuccess] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const providers: ProviderInfo[] = useMemo(() => {
    return settings?.providers && settings.providers.length > 0
      ? settings.providers
      : FALLBACK_PROVIDERS
  }, [settings])

  const selectedProviderInfo = useMemo(() => {
    return providers.find((p) => p.id === currentTab) || providers[0]
  }, [providers, currentTab])

  useEffect(() => {
    if (!open) return
    let active = true

    async function load() {
      setLoading(true)
      setError(null)
      setSaveSuccess(false)
      setTestResults({})
      try {
        const s = await fetchSettings()
        if (!active) return
        setSettings(s)
        const prov = s.llm_provider || 'gemini'
        setActiveProvider(prov)
        setCurrentTab(prov)
        setActiveModel(s.llm_model || 'gemini-3.1-flash-lite')
        setEnableLlm(s.enable_llm)
        setFlorenceDefault(s.florence_default)
        setOpenaiBaseUrl(s.openai_base_url || '')
        setKeysInput((prev) => ({
          ...prev,
          ollama: s.ollama_base_url || 'http://localhost:11434',
        }))
      } catch (err) {
        if (!active) return
        setError(err instanceof Error ? err.message : 'Failed to load settings from server.')
      } finally {
        if (active) setLoading(false)
      }
    }

    void load()

    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => {
      active = false
      window.removeEventListener('keydown', onKey)
    }
  }, [open, onClose])

  async function handleTest(providerId: string) {
    setTesting(true)
    setError(null)
    setTestResults((prev) => ({ ...prev, [providerId]: undefined as unknown as { valid: boolean; message: string } }))

    const provInfo = providers.find((p) => p.id === providerId)
    const currentModel =
      activeProvider === providerId ? activeModel : provInfo?.default_model || ''
    const typedKey = keysInput[providerId]?.trim() || ''

    try {
      const res = await testApiKey({
        provider: providerId,
        apiKey: typedKey || undefined,
        model: currentModel || undefined,
        baseUrl:
          providerId === 'ollama'
            ? typedKey || 'http://localhost:11434'
            : providerId === 'openai'
              ? openaiBaseUrl || undefined
              : undefined,
      })
      setTestResults((prev) => ({ ...prev, [providerId]: res }))
    } catch (err) {
      setTestResults((prev) => ({
        ...prev,
        [providerId]: {
          valid: false,
          message: err instanceof Error ? err.message : 'Connection probe failed.',
        },
      }))
    } finally {
      setTesting(false)
    }
  }

  async function handleSave() {
    setSaving(true)
    setError(null)
    setSaveSuccess(false)
    try {
      const payload: SettingsUpdatePayload = {
        llm_provider: activeProvider,
        llm_model: activeModel,
        enable_llm: enableLlm,
        florence_default: florenceDefault,
      }

      if (keysInput.gemini?.trim()) payload.gemini_api_key = keysInput.gemini.trim()
      if (keysInput.openai?.trim()) payload.openai_api_key = keysInput.openai.trim()
      if (openaiBaseUrl.trim()) payload.openai_base_url = openaiBaseUrl.trim()
      if (keysInput.anthropic?.trim()) payload.anthropic_api_key = keysInput.anthropic.trim()
      if (keysInput.groq?.trim()) payload.groq_api_key = keysInput.groq.trim()
      if (keysInput.openrouter?.trim()) payload.openrouter_api_key = keysInput.openrouter.trim()
      if (keysInput.ollama?.trim()) payload.ollama_base_url = keysInput.ollama.trim()

      const updated = await updateSettings(payload)
      setSettings(updated)

      // Clear typed sensitive inputs after successful persistence
      setKeysInput((prev) => ({
        ...prev,
        gemini: '',
        openai: '',
        anthropic: '',
        groq: '',
        openrouter: '',
      }))

      setSaveSuccess(true)
      if (onSettingsSaved) onSettingsSaved()
      setTimeout(() => setSaveSuccess(false), 4000)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings.')
    } finally {
      setSaving(false)
    }
  }

  function handleProviderChange(newProv: string) {
    setActiveProvider(newProv)
    setCurrentTab(newProv)
    const provInfo = providers.find((p) => p.id === newProv)
    if (provInfo && provInfo.models && provInfo.models.length > 0) {
      // Pick default model for this provider if current model doesn't belong to it
      const hasModel = provInfo.models.some((m) => m.id === activeModel)
      if (!hasModel) {
        setActiveModel(provInfo.default_model)
      }
    }
  }

  if (!open) return null

  return (
    <div className="credits-backdrop" onClick={onClose} role="presentation">
      <div
        className="credits-dialog panel settings-dialog"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        style={{ maxWidth: '680px', width: '94%' }}
      >
        <div className="credits-head">
          <div>
            <p className="eyebrow">Platform Configuration</p>
            <h2 className="credits-title" id="settings-title">
              AI Providers &amp; Clinical Models
            </h2>
          </div>
          <button
            type="button"
            className="button-quiet button-mini"
            onClick={onClose}
            aria-label="Close settings"
          >
            Close
          </button>
        </div>

        <div className="credits-body" style={{ maxHeight: '80vh', overflowY: 'auto' }}>
          {loading ? (
            <p style={{ padding: '32px 0', textAlign: 'center', color: 'var(--ink-subtle)' }}>
              Loading platform configuration and AI provider status...
            </p>
          ) : (
            <>
              {error && (
                <div
                  style={{
                    padding: '10px 14px',
                    marginBottom: '16px',
                    background: 'var(--red-tint, #fee2e2)',
                    color: 'var(--red-ink, #991b1b)',
                    borderRadius: '8px',
                    fontSize: '13px',
                  }}
                >
                  {error}
                </div>
              )}

              {saveSuccess && (
                <div
                  style={{
                    padding: '10px 14px',
                    marginBottom: '16px',
                    background: 'var(--green-tint, #dcfce7)',
                    color: 'var(--green-ink, #166534)',
                    borderRadius: '8px',
                    fontSize: '13px',
                    fontWeight: 500,
                  }}
                >
                  Settings updated successfully! Changes applied to active session and saved to .env.
                </div>
              )}

              {/* ACTIVE PROVIDER BAR */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '12px 14px',
                  borderRadius: '8px',
                  background: 'var(--surface-subtle, #f8fafc)',
                  border: '1px solid var(--border-color, #e2e8f0)',
                  marginBottom: '18px',
                }}
              >
                <div>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--ink-subtle)', fontWeight: 600 }}>
                    Active Diagnostic Provider
                  </div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--ink-primary, #0f172a)', marginTop: '2px' }}>
                    {providers.find((p) => p.id === activeProvider)?.name || activeProvider}
                    <span style={{ fontSize: '12px', fontWeight: 400, color: 'var(--ink-subtle)', marginLeft: '8px' }}>
                      ({activeModel})
                    </span>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {providers.find((p) => p.id === activeProvider)?.configured ? (
                    <span className="tag is-green" style={{ fontSize: '11px' }}>
                      <span className="tag-dot" />
                      Ready
                    </span>
                  ) : (
                    <span className="tag is-red" style={{ fontSize: '11px' }}>
                      <span className="tag-dot" />
                      Key Required
                    </span>
                  )}
                </div>
              </div>

              {/* PROVIDER TABS SELECTOR */}
              <div style={{ marginBottom: '16px' }}>
                <p className="credits-label" style={{ marginBottom: '8px' }}>
                  Supported AI Providers
                </p>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '6px',
                    marginBottom: '12px',
                  }}
                >
                  {providers.map((p) => {
                    const isSelected = currentTab === p.id
                    const isActive = activeProvider === p.id
                    return (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => setCurrentTab(p.id)}
                        className={`tag tag-button ${isSelected ? 'is-blue' : ''}`}
                        style={{
                          padding: '6px 12px',
                          fontSize: '12px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '6px',
                          border: isSelected ? '1px solid var(--accent, #2563eb)' : '1px solid var(--border-color, #cbd5e1)',
                          fontWeight: isSelected ? 600 : 400,
                          cursor: 'pointer',
                        }}
                      >
                        <span>{p.name.split(' ')[0]}</span>
                        {isActive && (
                          <span
                            style={{
                              fontSize: '9px',
                              textTransform: 'uppercase',
                              padding: '1px 4px',
                              borderRadius: '4px',
                              background: isSelected ? '#ffffff' : 'var(--blue-tint, #dbeafe)',
                              color: isSelected ? 'var(--accent, #2563eb)' : '#1e40af',
                              fontWeight: 700,
                            }}
                          >
                            Active
                          </span>
                        )}
                        {p.configured ? (
                          <span style={{ color: isSelected ? '#ffffff' : '#16a34a', fontSize: '11px' }}>●</span>
                        ) : (
                          <span style={{ color: isSelected ? '#ffffff' : '#94a3b8', fontSize: '11px' }}>○</span>
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* SELECTED PROVIDER CONFIGURATION CARD */}
              <div
                className="credits-section"
                style={{
                  border: '1px solid var(--border-color, #e2e8f0)',
                  borderRadius: '8px',
                  padding: '14px 16px',
                  marginBottom: '20px',
                  background: 'var(--bg-canvas, #ffffff)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <div>
                    <h4 style={{ margin: 0, fontSize: '14px', fontWeight: 600, color: 'var(--ink-primary, #0f172a)' }}>
                      {selectedProviderInfo.name}
                    </h4>
                    <p style={{ margin: '3px 0 0', fontSize: '12px', color: 'var(--ink-subtle)' }}>
                      {selectedProviderInfo.description}
                    </p>
                  </div>
                  {activeProvider !== selectedProviderInfo.id ? (
                    <button
                      type="button"
                      className="tag tag-button is-blue"
                      onClick={() => handleProviderChange(selectedProviderInfo.id)}
                      style={{ fontSize: '11px', padding: '4px 10px', fontWeight: 600 }}
                    >
                      Make Active Provider
                    </button>
                  ) : (
                    <span className="tag is-green" style={{ fontSize: '11px' }}>
                      <span className="tag-dot" />
                      Active Provider
                    </span>
                  )}
                </div>

                {/* Credential or URL Input */}
                {selectedProviderInfo.requires_key ? (
                  <div style={{ marginTop: '12px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--ink-primary)' }}>
                      API Key ({selectedProviderInfo.env_key})
                    </label>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
                      <input
                        type={showKeys[selectedProviderInfo.id] ? 'text' : 'password'}
                        className="settings-input"
                        placeholder={
                          selectedProviderInfo.configured
                            ? `Configured (${selectedProviderInfo.key_masked})`
                            : `Paste ${selectedProviderInfo.name} API key...`
                        }
                        value={keysInput[selectedProviderInfo.id] || ''}
                        onChange={(e) => {
                          const val = e.target.value
                          setKeysInput((prev) => ({ ...prev, [selectedProviderInfo.id]: val }))
                          setTestResults((prev) => ({ ...prev, [selectedProviderInfo.id]: undefined as unknown as { valid: boolean; message: string } }))
                        }}
                        style={{
                          flex: 1,
                          padding: '8px 12px',
                          fontSize: '13px',
                          borderRadius: '6px',
                          border: '1px solid var(--border-color, #cbd5e1)',
                          background: 'var(--bg-canvas, #ffffff)',
                          color: 'var(--ink-primary, #0f172a)',
                        }}
                      />
                      <button
                        type="button"
                        className="tag tag-button"
                        onClick={() =>
                          setShowKeys((prev) => ({
                            ...prev,
                            [selectedProviderInfo.id]: !prev[selectedProviderInfo.id],
                          }))
                        }
                        style={{ minWidth: '55px' }}
                      >
                        {showKeys[selectedProviderInfo.id] ? 'Hide' : 'Show'}
                      </button>
                      <button
                        type="button"
                        className="tag tag-button is-blue"
                        onClick={() => void handleTest(selectedProviderInfo.id)}
                        disabled={
                          testing ||
                          (!keysInput[selectedProviderInfo.id]?.trim() &&
                            !selectedProviderInfo.configured)
                        }
                        style={{ minWidth: '85px', fontWeight: 600 }}
                      >
                        {testing ? 'Testing...' : 'Test Key'}
                      </button>
                    </div>

                    {selectedProviderInfo.key_url && (
                      <div style={{ fontSize: '11px', color: 'var(--ink-subtle)', marginTop: '4px' }}>
                        Need an API key? Get one from{' '}
                        <a
                          href={selectedProviderInfo.key_url}
                          target="_blank"
                          rel="noreferrer noopener"
                          className="credits-link"
                        >
                          {selectedProviderInfo.name} Dashboard
                        </a>
                      </div>
                    )}
                  </div>
                ) : selectedProviderInfo.id === 'ollama' ? (
                  <div style={{ marginTop: '12px' }}>
                    <label style={{ display: 'block', fontSize: '12px', fontWeight: 500, marginBottom: '4px', color: 'var(--ink-primary)' }}>
                      Ollama Endpoint Base URL
                    </label>
                    <div style={{ display: 'flex', gap: '8px', marginBottom: '6px' }}>
                      <input
                        type="text"
                        className="settings-input"
                        placeholder="http://localhost:11434"
                        value={keysInput.ollama || ''}
                        onChange={(e) => {
                          const val = e.target.value
                          setKeysInput((prev) => ({ ...prev, ollama: val }))
                        }}
                        style={{
                          flex: 1,
                          padding: '8px 12px',
                          fontSize: '13px',
                          borderRadius: '6px',
                          border: '1px solid var(--border-color, #cbd5e1)',
                          background: 'var(--bg-canvas, #ffffff)',
                          color: 'var(--ink-primary, #0f172a)',
                        }}
                      />
                      <button
                        type="button"
                        className="tag tag-button is-blue"
                        onClick={() => void handleTest('ollama')}
                        disabled={testing}
                        style={{ minWidth: '85px', fontWeight: 600 }}
                      >
                        {testing ? 'Testing...' : 'Test Host'}
                      </button>
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--ink-subtle)' }}>
                      Local server for offline inference. Example models:{' '}
                      <code>ollama run medgemma</code> or <code>ollama run llama3.2</code>
                    </div>
                  </div>
                ) : (
                  <div style={{ marginTop: '12px', padding: '8px 12px', background: 'var(--surface-subtle, #f8fafc)', borderRadius: '6px', fontSize: '12px', color: 'var(--ink-subtle)' }}>
                    Fully offline embedded HuggingFace model. No external network credentials required.
                  </div>
                )}

                {/* Optional Custom OpenAI Base URL */}
                {selectedProviderInfo.id === 'openai' && (
                  <div style={{ marginTop: '10px' }}>
                    <label style={{ display: 'block', fontSize: '11px', color: 'var(--ink-subtle)', marginBottom: '3px' }}>
                      Custom OpenAI Base URL (Optional, for proxies, Azure, or LocalAI)
                    </label>
                    <input
                      type="text"
                      className="settings-input"
                      placeholder="https://api.openai.com/v1"
                      value={openaiBaseUrl}
                      onChange={(e) => setOpenaiBaseUrl(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '6px 10px',
                        fontSize: '12px',
                        borderRadius: '6px',
                        border: '1px solid var(--border-color, #cbd5e1)',
                        background: 'var(--bg-canvas, #ffffff)',
                        color: 'var(--ink-primary, #0f172a)',
                      }}
                    />
                  </div>
                )}

                {/* Test Connection Probe Feedback */}
                {testResults[selectedProviderInfo.id] && (
                  <div
                    style={{
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '12px',
                      marginTop: '8px',
                      background: testResults[selectedProviderInfo.id].valid
                        ? 'var(--green-tint, #dcfce7)'
                        : 'var(--red-tint, #fee2e2)',
                      color: testResults[selectedProviderInfo.id].valid
                        ? 'var(--green-ink, #166534)'
                        : 'var(--red-ink, #991b1b)',
                    }}
                  >
                    {testResults[selectedProviderInfo.id].valid ? '✓ ' : '✕ '}
                    {testResults[selectedProviderInfo.id].message}
                  </div>
                )}
              </div>

              {/* MODEL SELECTION FOR ACTIVE PROVIDER */}
              <section className="credits-section" style={{ marginBottom: '20px' }}>
                <p className="credits-label" style={{ marginBottom: '6px' }}>
                  Model Selection for {providers.find((p) => p.id === activeProvider)?.name}
                </p>
                <div style={{ marginBottom: '12px' }}>
                  <select
                    className="settings-select"
                    value={activeModel}
                    onChange={(e) => setActiveModel(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '8px 12px',
                      fontSize: '13px',
                      borderRadius: '6px',
                      border: '1px solid var(--border-color, #cbd5e1)',
                      background: 'var(--bg-canvas, #ffffff)',
                      color: 'var(--ink-primary, #0f172a)',
                    }}
                  >
                    {(
                      providers.find((p) => p.id === activeProvider)?.models || []
                    ).map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </section>

              {/* DIAGNOSTIC TOGGLES */}
              <section className="credits-section" style={{ marginBottom: '20px' }}>
                <p className="credits-label" style={{ marginBottom: '8px' }}>
                  Diagnostic Pipeline Defaults
                </p>

                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 0',
                    borderBottom: '1px solid var(--border-color, #e2e8f0)',
                  }}
                >
                  <div>
                    <p style={{ fontSize: '13px', fontWeight: 500, margin: 0 }}>
                      Enable Clinical AI Assistant
                    </p>
                    <p style={{ fontSize: '11px', color: 'var(--ink-subtle)', margin: '2px 0 0' }}>
                      Show conversational copilot drawer and enable diagnostic report smoothing
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={enableLlm}
                    onChange={(e) => setEnableLlm(e.target.checked)}
                    style={{ transform: 'scale(1.2)', cursor: 'pointer' }}
                  />
                </div>

                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 0',
                  }}
                >
                  <div>
                    <p style={{ fontSize: '13px', fontWeight: 500, margin: 0 }}>
                      Enable Florence-2 VLM by Default
                    </p>
                    <p style={{ fontSize: '11px', color: 'var(--ink-subtle)', margin: '2px 0 0' }}>
                      Inspect visual raster plots for recruitment patterns (Adds ~15s on CPU)
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={florenceDefault}
                    onChange={(e) => setFlorenceDefault(e.target.checked)}
                    style={{ transform: 'scale(1.2)', cursor: 'pointer' }}
                  />
                </div>
              </section>

              {/* MODAL FOOTER */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: '10px',
                  paddingTop: '16px',
                  borderTop: '1px solid var(--border-color, #e2e8f0)',
                }}
              >
                <button type="button" className="tag tag-button" onClick={onClose}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="tag tag-button is-green"
                  onClick={() => void handleSave()}
                  disabled={saving}
                  style={{ fontWeight: 600, padding: '6px 18px' }}
                >
                  {saving ? 'Saving Changes...' : 'Save & Apply'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
