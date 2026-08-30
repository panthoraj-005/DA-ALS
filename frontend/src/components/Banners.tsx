/** Persistent, never dismissible. It is on screen for every result. */
export function DisclaimerBar({ text }: { text: string }) {
  return (
    <div className="notice is-diagnosis" role="note">
      <div className="notice-inner">
        <span className="notice-label">Not a diagnosis</span>
        <span>{text}</span>
      </div>
    </div>
  )
}

export function DemoBanner() {
  return (
    <div className="notice is-demo" role="alert">
      <div className="notice-inner">
        <span className="notice-label">Demo mode</span>
        <span>
          The server is running on untrained placeholder weights, so every number on this page
          is noise. Put the trained artifacts in <code>models/</code> and restart without{' '}
          <code>DEMO_MODE</code> to get real results.
        </span>
      </div>
    </div>
  )
}
