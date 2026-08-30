import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Top-level boundary. A render crash must never leave a clinician staring at a
 * blank page with a stale verdict still in their head — say what happened, keep
 * the disclaimer visible, and offer the one action that fixes it.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Interface error:', error, info.componentStack)
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div className="above">
        <div className="notice is-diagnosis" role="note">
          <div className="notice-inner">
            <span className="notice-label">Not a diagnosis</span>
            <span>
              Nothing on this screen is a medical determination. Any result shown before this
              error should be confirmed by re-running the signal.
            </span>
          </div>
        </div>

        <section className="panel crash" role="alert">
          <p className="eyebrow">Interface error</p>
          <h1 className="crash-title">This screen stopped rendering</h1>
          <p className="crash-copy">
            The screening backend is unaffected — this is a fault in the browser interface.
            Reloading restores it. Your session history is not saved, so any previous results
            will need re-running.
          </p>
          <p className="crash-detail">{error.message || String(error)}</p>
          <button type="button" className="button-quiet" onClick={() => window.location.reload()}>
            Reload the interface
          </button>
        </section>
      </div>
    )
  }
}
