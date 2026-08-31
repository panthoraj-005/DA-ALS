import type { Health } from '../types'

const STATUS_COPY: Record<Health['status'], { text: string; tone: string }> = {
  ready: { text: 'All models loaded', tone: 'is-green' },
  degraded: { text: 'Partial · CNN only', tone: 'is-yellow' },
  demo: { text: 'Demo weights', tone: 'is-yellow' },
  unavailable: { text: 'No models loaded', tone: 'is-red' },
}

interface Props {
  health: Health | null
  offline: boolean
  onToggleDetails: () => void
  detailsOpen: boolean
  onToggleChat?: () => void
  chatOpen?: boolean
  onOpenCredits: () => void
}

export default function Masthead({
  health,
  offline,
  onToggleDetails,
  detailsOpen,
  onToggleChat,
  chatOpen,
  onOpenCredits,
}: Props) {
  const status = health ? STATUS_COPY[health.status] : null

  return (
    <header className="masthead">
      <div className="masthead-inner">
        <div className="wordmark">
          <svg className="glyph" viewBox="0 0 30 30" aria-hidden="true" fill="none">
            <path
              d="M2 17.5h4.4l1.8-6.2 2.3 11.4L13.1 6l2.4 18 2.2-11.2 1.6 4.7h6.7"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </svg>
          <div>
            <h1 className="wordmark-title">EMG ALS screening</h1>
            <p className="wordmark-sub">CNN · Florence-2 · XGBoost</p>
          </div>
        </div>

        <div className="status-cluster">
          {offline && (
            <span className="tag is-red">
              <span className="tag-dot" />
              Server unreachable
            </span>
          )}
          {status && (
            <span className={`tag ${status.tone}`}>
              <span className="tag-dot" />
              {status.text}
            </span>
          )}
          {health && <span className="tag">{health.device}</span>}
          <button type="button" className="tag tag-button" onClick={onToggleDetails}>
            {detailsOpen ? 'Hide status' : 'Model status'}
          </button>
          <button type="button" className="tag tag-button" onClick={onOpenCredits}>
            Credits
          </button>
          {onToggleChat && (
            <button
              type="button"
              className={`tag tag-button ${chatOpen ? 'is-blue' : ''}`}
              onClick={onToggleChat}
              style={{ fontWeight: 600 }}
            >
              <span className="tag-dot" style={{ background: 'var(--blue-ink)' }} />
              {chatOpen ? 'Close Assistant' : 'AI Assistant'}
            </button>
          )}
        </div>
      </div>
    </header>
  )
}
