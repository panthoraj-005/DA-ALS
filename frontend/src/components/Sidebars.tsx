import type { Health, PredictionResult } from '../types'

export function HealthPanel({ health }: { health: Health | null }) {
  if (!health) return null

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Model status</h2>
        <p className="eyebrow num">fs {health.config.fs} Hz</p>
      </div>
      <div className="panel-body">
        <ul className="artifact-list">
          {health.artifacts.map((artifact) => (
            <li className="artifact-row" key={artifact.name}>
              <span className={`artifact-mark${artifact.loaded ? ' is-on' : ''}`} />
              <div>
                <p className="artifact-name">{artifact.name}</p>
                <p className="artifact-detail">{artifact.detail}</p>
              </div>
            </li>
          ))}
        </ul>
        <p className="footnote">
          Expecting {health.config.signal_length.toLocaleString()} points per signal, bandpass{' '}
          {health.config.bandpass[0]}–{health.config.bandpass[1]} Hz,{' '}
          {health.config.n_segments} segments, {health.config.meta_feature_dim}-D fusion vector.
        </p>
      </div>
    </section>
  )
}

interface HistoryProps {
  items: PredictionResult[]
  currentId: string | null
  onSelect: (result: PredictionResult) => void
}

export function History({ items, currentId, onSelect }: HistoryProps) {
  if (items.length === 0) return null

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">This session</h2>
        <p className="eyebrow num">{items.length}</p>
      </div>
      <div className="panel-body">
        <ul className="history-list">
          {items.map((item) => {
            const isAls = item.final_prediction === 'ALS'
            return (
              <li
                className={`history-item${item.id === currentId ? ' is-current' : ''}`}
                key={item.id}
              >
                <button type="button" className="history-button" onClick={() => onSelect(item)}>
                  <div style={{ minWidth: 0 }}>
                    <p className="history-source">{item.source}</p>
                    <p className="history-sub">
                      {(item.final_confidence * 100).toFixed(1)}% · {item.fusion}
                    </p>
                  </div>
                  <span className={`tag ${isAls ? 'is-red' : 'is-green'}`}>
                    {item.final_prediction}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </section>
  )
}

export function EmptyStage({ health, offline }: { health: Health | null; offline: boolean }) {
  const blocked = health?.status === 'unavailable'

  let title = 'No signal screened yet'
  let copy =
    'Drop an EMG recording on the left or pick a bundled sample, then run the screening. ' +
    'The result shows every model’s reading, not just the final label.'

  if (offline) {
    title = 'Waiting for the server'
    copy =
      'The screening backend is not responding. Start it with `uvicorn main:app --port 8000` ' +
      'in the backend folder, or `docker compose up`, then run a signal.'
  } else if (blocked) {
    title = 'No models are loaded'
    copy =
      'The server started without a CNN checkpoint. Put the trained artifacts in the models ' +
      'folder and restart, or start the backend with DEMO_MODE=1 to walk through the interface.'
  }

  return (
    <section className="panel stage-empty">
      <svg width="160" height="40" viewBox="0 0 160 40" fill="none" aria-hidden="true">
        <path
          d="M2 20h18l4-11 5 24 6-32 5 38 5-25 4 11h17l4-7 4 14 4-11h16l4-13 5 26 5-20 4 6h19"
          stroke="#dcdcd9"
          strokeWidth="1.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
      <h2 className="stage-empty-title">{title}</h2>
      <p className="stage-empty-copy">{copy}</p>
    </section>
  )
}

export function Working({ withVlm }: { withVlm: boolean }) {
  return (
    <section className="panel" aria-live="polite">
      <div className="working">
        <span>{withVlm ? 'CNN · vision model · fusion' : 'Running the CNN'}</span>
        <span className="working-bar" />
      </div>
    </section>
  )
}

interface ErrorStageProps {
  error: string
  onClear: () => void
  onSelectSample: (name: string) => void
  samples: { name: string }[]
}

export function ErrorStage({ error, onClear, onSelectSample, samples }: ErrorStageProps) {
  const isPickle = /pickle|pandas|unpickled|numeric/i.test(error)
  const isLength = /point|length|23,437|23437/i.test(error)
  const isMultiDim = /shape|column|row|\d+x\d+/i.test(error)

  return (
    <section className="panel stage-error">
      <div className="error-stage-head">
        <div className="error-badge-row">
          <span className="tag is-red">
            <span className="tag-dot" /> Screening Failed
          </span>
          <button type="button" className="button-quiet button-mini" onClick={onClear}>
            Clear error
          </button>
        </div>
        <h2 className="error-stage-title">Signal Processing Error</h2>
        <div className="error-callout">
          <p className="error-callout-text">{error}</p>
        </div>
      </div>

      <div className="error-stage-body">
        <h3 className="error-subheading">Diagnostic & Recommendations</h3>

        {isPickle && (
          <div className="suggestion-box">
            <div className="suggestion-badge">Format Mismatch</div>
            <p className="suggestion-desc">
              The uploaded file contains labels or legacy Python/pandas structures that cannot be loaded. 
              The screening pipeline requires raw 1-D numerical voltage readings.
            </p>
            <div className="code-example">
              <span className="code-example-title">Export your continuous EMG waveform with NumPy:</span>
              <pre>
                <code>import numpy as np&#10;np.save("patient_signal.npy", signal_data.astype(np.float32))</code>
              </pre>
            </div>
          </div>
        )}

        {isLength && (
          <div className="suggestion-box">
            <div className="suggestion-badge">Dimension Requirement</div>
            <p className="suggestion-desc">
              The model architecture requires an exact waveform length of <strong>23,437 points</strong> (~23.4 seconds at 1,000 Hz).
              Please resample or trim the recording before uploading.
            </p>
          </div>
        )}

        {isMultiDim && (
          <div className="suggestion-box">
            <div className="suggestion-badge">Single Channel Required</div>
            <p className="suggestion-desc">
              The file contains multiple data columns or a matrix. Please upload a single-channel 1-D recording.
            </p>
          </div>
        )}

        <div className="suggestion-box">
          <div className="suggestion-badge">Supported Formats</div>
          <p className="suggestion-desc">
            <code>.npy</code> (NumPy 1-D array), <code>.csv</code> (single column of numbers), or <code>.txt</code>.
          </p>
        </div>

        {samples.length > 0 && (
          <div className="error-sample-pick">
            <p className="eyebrow" style={{ marginBottom: '8px' }}>
              Or run with a validated dataset sample:
            </p>
            <div className="sample-quick-list">
              {samples.slice(0, 4).map((s) => (
                <button
                  key={s.name}
                  type="button"
                  className="button-quiet button-mini"
                  onClick={() => {
                    onSelectSample(s.name)
                    onClear()
                  }}
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
