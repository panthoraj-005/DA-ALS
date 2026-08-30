import type { PredictionResult } from '../types'

function Cell({ label, value, quiet }: { label: string; value: string; quiet?: boolean }) {
  return (
    <div className="readout-cell">
      <p className="eyebrow">{label}</p>
      <p className={`readout-value${quiet ? ' is-quiet' : ''}`}>{value}</p>
    </div>
  )
}

export function Explanation({ result }: { result: PredictionResult }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Explanation</h2>
        <p className="eyebrow">{result.explanation_source ?? 'grounded template'}</p>
      </div>
      <div className="panel-body">
        <p className="explanation">{result.explanation}</p>

        {result.florence_caption && (
          <div className="caption-block">
            <p className="eyebrow" style={{ marginBottom: 6 }}>
              Vision model caption — unverified
            </p>
            {result.florence_caption}
          </div>
        )}
      </div>
    </section>
  )
}

export function Readout({ result }: { result: PredictionResult }) {
  const totalMs = result.timings.total_ms

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Readout</h2>
        <p className="eyebrow num">
          {totalMs ? `${(totalMs / 1000).toFixed(2)}s · ${result.device}` : result.device}
        </p>
      </div>

      <div className="panel-body">
        <div className="readout">
          <Cell label="CNN" value={result.cnn_probability.toFixed(4)} />
          <Cell
            label="Florence-2"
            value={result.florence_probability?.toFixed(4) ?? 'not run'}
            quiet={result.florence_probability === null}
          />
          <Cell
            label="Fused"
            value={result.meta_probability?.toFixed(4) ?? 'not run'}
            quiet={result.meta_probability === null}
          />
          <Cell label="Abnormal segments" value={result.abnormal_segments} />
          <Cell label="Severity" value={result.severity} quiet={result.severity === 'N/A'} />
          <Cell label="Points" value={result.signal_length.toLocaleString()} />
        </div>
      </div>
    </section>
  )
}

export function Attribution({ result }: { result: PredictionResult }) {
  const shap = result.top_shap_features ?? []
  const maxAbs = shap.reduce((m, f) => Math.max(m, Math.abs(f.value)), 0) || 1

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Attribution</h2>
        <p className="eyebrow">SHAP</p>
      </div>

      <div className="panel-body">
        {shap.length > 0 ? (
          <>
            <p className="footnote" style={{ marginTop: 0 }}>
              What moved the fused decision, and in which direction.
            </p>
            <ul className="shap-list">
              {shap.map((feature) => {
                const width = (Math.abs(feature.value) / maxAbs) * 50
                const toAls = feature.value > 0
                return (
                  <li className="shap-row" key={feature.feature}>
                    <div>
                      <p className="shap-name">{feature.feature}</p>
                      <p className="shap-label">
                        {feature.label} · {feature.direction}
                      </p>
                    </div>
                    <div className="shap-bar" title={feature.value.toFixed(5)}>
                      <span className="shap-bar-mid" />
                      <span
                        className={`shap-bar-fill${toAls ? '' : ' is-normal'}`}
                        style={
                          toAls
                            ? { left: '50%', width: `${width}%` }
                            : { right: '50%', width: `${width}%` }
                        }
                      />
                    </div>
                  </li>
                )
              })}
            </ul>
          </>
        ) : (
          <p className="footnote" style={{ marginTop: 0 }}>
            SHAP attribution needs the fused meta-learner. Run the vision model with the
            meta-learner loaded to see which input moved the decision.
          </p>
        )}
      </div>
    </section>
  )
}
