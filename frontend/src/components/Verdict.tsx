import { reportUrl } from '../api'
import type { PredictionResult } from '../types'

const FUSION_COPY: Record<string, string> = {
  cnn_only: 'the CNN alone, on the fast path',
  'cnn+florence': 'the CNN, with the vision model unfused',
  'cnn+florence+meta': 'the CNN and Florence-2, fused by XGBoost',
}

function DownloadIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path
        d="M7 1.5v8m0 0L4 6.7M7 9.5l3-2.8M1.8 11.2h10.4"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function Verdict({ result }: { result: PredictionResult }) {
  const isAls = result.final_prediction === 'ALS'
  const pct = result.final_confidence * 100

  return (
    <section className={`panel verdict ${isAls ? 'is-als' : 'is-normal'}`}>
      <div className="verdict-top">
        <div>
          <p className="eyebrow">Screening result</p>
          <p className="verdict-word">{result.final_prediction}</p>
          <p className="verdict-meta">
            {result.severity !== 'N/A' && (
              <>
                Severity <strong>{result.severity}</strong>. </>
            )}
            Decided by {FUSION_COPY[result.fusion] ?? result.fusion}. Record{' '}
            <span className="num">{result.id}</span>.
          </p>
        </div>

        <div className="verdict-actions">
          <a
            className="button-quiet"
            href={reportUrl(result.id)}
            target="_blank"
            rel="noreferrer"
          >
            <DownloadIcon />
            PDF report
          </a>
        </div>
      </div>

      <div className="meter">
        <div
          className="meter-scale"
          role="meter"
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Confidence in the final reading"
        >
          <div className="meter-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="meter-ticks">
          <span>confidence</span>
          <strong className="num">{pct.toFixed(1)}%</strong>
        </div>
      </div>
    </section>
  )
}
