import type { PredictionResult } from '../types'

/**
 * The signature element.
 *
 * Every model that had a say gets the same axis: ALS probability from 0 to 1,
 * with the 0.5 decision boundary drawn as a hard line. Putting them on one
 * shared scale is the whole point — you can see at a glance not just what each
 * model said but how far from the boundary it said it, and where they part
 * company. A stack of three separate percentages cannot show that.
 */

interface AxisProps {
  name: string
  role: string
  probability: number | null
  emphasis?: boolean
  emptyNote?: string
}

function Axis({ name, role, probability, emphasis, emptyNote }: AxisProps) {
  const hasValue = probability !== null && Number.isFinite(probability)
  const pct = hasValue ? Math.min(Math.max(probability as number, 0), 1) * 100 : 50
  const isAls = hasValue && (probability as number) > 0.5
  const flip = pct > 62

  return (
    <div className="rail-axis">
      <div className="rail-label">
        <p className="rail-name">{name}</p>
        <p className="rail-role">{role}</p>
      </div>

      <div
        className="rail-track"
        role="img"
        aria-label={
          hasValue
            ? `${name}: ALS probability ${(pct / 100).toFixed(3)}, reading ${isAls ? 'ALS' : 'Normal'}`
            : `${name}: ${emptyNote ?? 'not run'}`
        }
      >
        <span className="rail-half" />
        <span className="rail-half rail-half-right" />
        <span className="rail-boundary" />

        {hasValue ? (
          <>
            <span
              className={`rail-marker ${isAls ? 'is-als' : 'is-normal'}${
                emphasis ? ' is-final' : ''
              }`}
              style={{ left: `${pct}%` }}
            />
            <span
              className={`rail-value ${flip ? 'is-flip' : 'is-normalside'}`}
              style={{ left: `${pct}%` }}
            >
              {(probability as number).toFixed(3)}
            </span>
          </>
        ) : (
          <span className="rail-empty">{emptyNote ?? 'not run'}</span>
        )}
      </div>

      <div className="rail-scale">
        <span>0.0 normal</span>
        <span>0.5</span>
        <span>als 1.0</span>
      </div>
    </div>
  )
}

export default function EvidenceRail({ result }: { result: PredictionResult }) {
  const agreementCopy =
    result.models_agree === null
      ? 'Only one model ran, so there is nothing to cross-check.'
      : result.models_agree
        ? 'The CNN and the vision model landed on the same side of the boundary.'
        : 'The CNN and the vision model disagree — treat this result as low-certainty.'

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Evidence</h2>
        <p className="eyebrow">{result.fusion.replace(/_/g, ' ').replace(/\+/g, ' + ')}</p>
      </div>

      <div className="panel-body">
        <div className="rail">
          <Axis
            name="1-D CNN"
            role="reads the waveform"
            probability={result.cnn_probability}
          />
          <Axis
            name="Florence-2"
            role="reads the plotted image"
            probability={result.florence_probability}
            emptyNote="not run"
          />
          <Axis
            name="Meta-learner"
            role="XGBoost fusion"
            probability={result.meta_probability}
            emphasis
            emptyNote="not fused"
          />
        </div>

        <p className="rail-foot">
          <span
            className={`tag ${
              result.models_agree === false
                ? 'is-red'
                : result.models_agree === true
                  ? 'is-green'
                  : ''
            }`}
          >
            {result.models_agree === null
              ? 'single model'
              : result.models_agree
                ? 'models agree'
                : 'models disagree'}
          </span>
          <span>{agreementCopy}</span>
        </p>

        {result.fusion_note && <p className="plot-note">{result.fusion_note}</p>}
      </div>
    </section>
  )
}
