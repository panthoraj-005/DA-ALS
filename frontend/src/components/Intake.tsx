import { useRef, useState } from 'react'
import type { Health, SampleSignal } from '../types'

interface Props {
  samples: SampleSignal[]
  health: Health | null
  busy: boolean
  error: string | null
  onRun: (input: { file?: File; sample?: string; explain: boolean; useVlm: boolean; useLlm: boolean }) => void
  onClearError?: () => void
}

const ACCEPT = '.npy,.csv,.txt,.pkl,.pickle'

export default function Intake({ samples, health, busy, error, onRun, onClearError }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [sample, setSample] = useState('')
  const [explain, setExplain] = useState(true)
  const [useVlm, setUseVlm] = useState(false)
  const [useLlm, setUseLlm] = useState(true)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const vlmAvailable = health?.capabilities.florence ?? false
  const llmAvailable = health?.capabilities.llm_explanation ?? true
  const canRun = !busy && (file !== null || sample !== '')

  function take(next: File | null) {
    setFile(next)
    onClearError?.()
    if (next) setSample('')
  }

  function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!canRun) return
    onRun({ file: file ?? undefined, sample: sample || undefined, explain, useVlm, useLlm })
  }

  return (
    <form className="panel" onSubmit={submit}>
      <div className="panel-head">
        <h2 className="panel-title">Signal</h2>
        <p className="eyebrow num">
          {health?.config.signal_length.toLocaleString() ?? '—'} pts
        </p>
      </div>

      <div className="panel-body">
        <div
          className={`dropzone${dragging ? ' is-over' : ''}${file ? ' has-file' : ''}`}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            const dropped = e.dataTransfer.files?.[0]
            if (dropped) take(dropped)
          }}
          role="button"
          tabIndex={0}
          aria-label="Choose a signal file"
        >
          {file ? (
            <div className="dropzone-file">
              <div style={{ minWidth: 0 }}>
                <p className="dropzone-name">{file.name}</p>
                <p className="dropzone-hint">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
              <button
                type="button"
                className="button-quiet button-mini"
                onClick={(e) => {
                  e.stopPropagation()
                  take(null)
                  if (inputRef.current) inputRef.current.value = ''
                }}
              >
                Remove
              </button>
            </div>
          ) : (
            <>
              <p className="dropzone-title">Drop a signal here</p>
              <p className="dropzone-hint">npy · csv · pkl</p>
            </>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => take(e.target.files?.[0] ?? null)}
        />

        <div className="divider-or">or</div>

        <label className="field-label eyebrow" htmlFor="sample-select">
          Bundled sample
        </label>
        <select
          id="sample-select"
          className="select"
          value={sample}
          disabled={samples.length === 0}
          onChange={(e) => {
            setSample(e.target.value)
            onClearError?.()
            if (e.target.value) take(null)
          }}
        >
          <option value="">
            {samples.length ? 'Choose a sample…' : 'No samples on the server'}
          </option>
          {samples.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>

        <div className="options">
          <label className={`toggle-row${vlmAvailable ? '' : ' is-disabled'}`}>
            <input
              type="checkbox"
              checked={useVlm && vlmAvailable}
              disabled={!vlmAvailable}
              onChange={(e) => setUseVlm(e.target.checked)}
            />
            <span className="toggle-copy">
              <span className="toggle-title">Run the vision model and fuse</span>
              <span className="toggle-note">
                {vlmAvailable
                  ? 'Florence-2 reads the rendered trace and XGBoost fuses both models. Slower — seconds to tens of seconds on CPU.'
                  : 'Florence-2 is not loaded on this server, so only the CNN can run.'}
              </span>
            </span>
          </label>

          <label className="toggle-row">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
            />
            <span className="toggle-copy">
              <span className="toggle-title">Compute explainability</span>
              <span className="toggle-note">
                Grad-CAM overlay and SHAP attribution. Adds about a second.
              </span>
            </span>
          </label>

          <label className={`toggle-row${llmAvailable ? '' : ' is-disabled'}`}>
            <input
              type="checkbox"
              checked={useLlm && llmAvailable}
              disabled={!llmAvailable}
              onChange={(e) => setUseLlm(e.target.checked)}
            />
            <span className="toggle-copy">
              <span className="toggle-title">AI Clinical Summary (LLM)</span>
              <span className="toggle-note">
                {llmAvailable
                  ? 'Use Gemini LLM to synthesize a natural-language clinical explanation.'
                  : 'LLM explanation is disabled on this server.'}
              </span>
            </span>
          </label>
        </div>

        <button
          className="button"
          type="submit"
          disabled={!canRun}
          style={{ marginTop: 'var(--s5)' }}
        >
          {busy ? 'Screening…' : 'Run screening'}
        </button>

        {error && (
          <p className="error-note" role="alert">
            {error}
          </p>
        )}
      </div>
    </form>
  )
}
