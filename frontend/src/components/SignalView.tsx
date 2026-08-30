import { useState } from 'react'
import { assetUrl } from '../api'
import type { PredictionResult } from '../types'

type View = 'annotated' | 'signal'

export default function SignalView({ result }: { result: PredictionResult }) {
  const hasGradcam = Boolean(result.gradcam_image_url)
  const [view, setView] = useState<View>(hasGradcam ? 'annotated' : 'signal')
  const [failed, setFailed] = useState(false)

  const active =
    view === 'annotated' && hasGradcam ? result.gradcam_image_url : result.signal_image_url
  const src = assetUrl(active)

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Trace</h2>
        <div className="tabs">
          <button
            type="button"
            className={`tab${view === 'annotated' ? ' is-active' : ''}`}
            onClick={() => setView('annotated')}
            disabled={!hasGradcam}
          >
            Grad-CAM
          </button>
          <button
            type="button"
            className={`tab${view === 'signal' ? ' is-active' : ''}`}
            onClick={() => setView('signal')}
          >
            Preprocessed
          </button>
        </div>
      </div>

      <div className="panel-body">
        {src && !failed ? (
          <div className="plot-tray">
            <img
              className="plot"
              src={src}
              onError={() => setFailed(true)}
              alt={
                view === 'annotated'
                  ? 'EMG trace with the Grad-CAM heat overlay and abnormal segments shaded'
                  : 'Preprocessed EMG trace with abnormal segments shaded'
              }
            />
          </div>
        ) : (
          <p className="plot-note">
            {failed
              ? 'The plot could not be loaded. Generated images expire on the server — re-run the signal to rebuild it.'
              : 'No image was produced for this signal.'}
          </p>
        )}

        <p className="plot-note">
          {view === 'annotated'
            ? 'Orange density is where the CNN looked (Grad-CAM on conv3). Red spans are segments whose energy exceeded this recording’s own mean + 1 SD.'
            : 'Bandpass 5–450 Hz, then z-scored — this is the image the vision model reads. Red spans mark the high-energy segments.'}
          {view === 'annotated' && result.gradcam_peak_percent !== undefined && (
            <>
              {' '}
              Peak attention sits at{' '}
              <span className="num">{result.gradcam_peak_percent.toFixed(0)}%</span> through the
              recording.
            </>
          )}
        </p>
      </div>
    </section>
  )
}
