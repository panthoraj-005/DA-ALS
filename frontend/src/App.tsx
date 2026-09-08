import { useCallback, useEffect, useState } from 'react'
import { fetchHealth, fetchSamples, predict, type PredictOptions } from './api'
import Masthead from './components/Masthead'
import { DemoBanner, DisclaimerBar } from './components/Banners'
import Intake from './components/Intake'
import Reveal from './components/Reveal'
import Verdict from './components/Verdict'
import EvidenceRail from './components/EvidenceRail'
import SignalView from './components/SignalView'
import { Attribution, Explanation, Readout } from './components/Details'
import { EmptyStage, ErrorStage, HealthPanel, History, Working } from './components/Sidebars'
import Chatbot from './components/Chatbot'
import Credits from './components/Credits'
import SettingsModal from './components/SettingsModal'
import type { Health, PredictionResult, SampleSignal } from './types'

const FALLBACK_DISCLAIMER =
  'Research and screening aid only. This result is not a medical diagnosis and must be ' +
  'confirmed by a qualified clinician.'

// While the server is unreachable, keep checking — a backend that is still
// loading Florence-2 comes back on its own, and the UI should notice.
const RECONNECT_MS = 6_000

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [offline, setOffline] = useState(false)
  const [samples, setSamples] = useState<SampleSignal[]>([])
  const [statusOpen, setStatusOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [creditsOpen, setCreditsOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)

  const [result, setResult] = useState<PredictionResult | null>(null)
  const [history, setHistory] = useState<PredictionResult[]>([])
  const [busy, setBusy] = useState(false)
  const [busyWithVlm, setBusyWithVlm] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadStatus = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([fetchHealth(), fetchSamples()])
      setHealth(h)
      setSamples(s)
      setOffline(false)
      if (h.status === 'unavailable' || h.status === 'degraded' || h.demo_mode) {
        setStatusOpen(true)
      }
      return true
    } catch {
      setOffline(true)
      setHealth(null)
      return false
    }
  }, [])

  useEffect(() => {
    void loadStatus()
  }, [loadStatus])

  // Poll only while offline, and stop as soon as the server answers.
  useEffect(() => {
    if (!offline) return
    const timer = setInterval(() => void loadStatus(), RECONNECT_MS)
    return () => clearInterval(timer)
  }, [offline, loadStatus])

  async function run(input: PredictOptions) {
    setBusy(true)
    setBusyWithVlm(input.useVlm)
    setError(null)
    setResult(null)
    try {
      const next = await predict(input)
      setResult(next)
      setHistory((prev) => [next, ...prev.filter((r) => r.id !== next.id)].slice(0, 12))
    } catch (err) {
      setResult(null)
      setError(err instanceof Error ? err.message : 'The screening request failed.')
      void loadStatus()
    } finally {
      setBusy(false)
    }
  }

  const disclaimer = result?.disclaimer ?? health?.disclaimer ?? FALLBACK_DISCLAIMER
  const showDemo = Boolean(health?.demo_mode || result?.demo_mode)

  return (
    <>
      <div className="atmosphere" aria-hidden="true" />

      <div className="above">
        <Masthead
          health={health}
          offline={offline}
          detailsOpen={statusOpen}
          onToggleDetails={() => setStatusOpen((v) => !v)}
          chatOpen={chatOpen}
          onToggleChat={() => setChatOpen((v) => !v)}
          onOpenCredits={() => setCreditsOpen(true)}
          onOpenSettings={() => setSettingsOpen(true)}
        />
        {showDemo && <DemoBanner />}
        <DisclaimerBar text={disclaimer} />

        <Credits open={creditsOpen} onClose={() => setCreditsOpen(false)} />
        <SettingsModal
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          onSettingsSaved={() => void loadStatus()}
        />

        <Chatbot
          open={chatOpen}
          onClose={() => setChatOpen(false)}
          currentResult={result}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        <main className="shell">
          <div className="column column-rail">
            <Intake
              samples={samples}
              health={health}
              busy={busy}
              error={
                offline
                  ? 'Cannot reach the screening server. Retrying every few seconds — start the backend and this clears itself.'
                  : error
              }
              onRun={run}
              onClearError={() => setError(null)}
            />
            {statusOpen && <HealthPanel health={health} />}
            <History
              items={history}
              currentId={result?.id ?? null}
              onSelect={(item) => {
                setError(null)
                setResult(item)
              }}
            />
          </div>

          <div className="column">
            {busy && <Working withVlm={busyWithVlm} />}
            {!busy && error && (
              <ErrorStage
                error={error}
                onClear={() => setError(null)}
                onSelectSample={(sampleName) => run({ sample: sampleName, explain: true, useVlm: false })}
                samples={samples}
              />
            )}
            {!busy && !error && !result && <EmptyStage health={health} offline={offline} />}
            {result && !busy && !error && (
              <>
                <Reveal index={0} key={`${result.id}-verdict`}>
                  <Verdict result={result} />
                </Reveal>

                <Reveal index={1} key={`${result.id}-bento`}>
                  <div className="bento">
                    <EvidenceRail result={result} />
                    <div className="column">
                      <Readout result={result} />
                      <Attribution result={result} />
                    </div>
                  </div>
                </Reveal>

                <Reveal index={2} key={`${result.id}-trace`}>
                  <SignalView result={result} />
                </Reveal>

                <Reveal index={3} key={`${result.id}-explain`}>
                  <Explanation result={result} />
                </Reveal>
              </>
            )}
          </div>
        </main>

        <footer className="footer">
          <div className="footer-inner">
            <p>
              Screening aid for research use. Every prediction here comes from an experimental
              model and is not a clinical determination.
            </p>
            <p className="num">
              CNN · Florence-2 · XGBoost ·{' '}
              <button
                type="button"
                className="credits-link credits-link-button"
                onClick={() => setCreditsOpen(true)}
                aria-haspopup="dialog"
              >
                Credits
              </button>
            </p>
          </div>
        </footer>

        {/* Floating Credits Button */}
        <button
          type="button"
          className="credits-popup-btn"
          onClick={() => setCreditsOpen(true)}
          aria-haspopup="dialog"
          title="Authors, tooling, and licence"
        >
          <svg className="credits-popup-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M12 16v-4M12 8h.01" />
          </svg>
          <span className="credits-popup-label">Credits</span>
        </button>

        {/* Floating AI Popup Button */}
        <button
          type="button"
          className={`chat-popup-btn ${chatOpen ? 'is-active' : ''}`}
          onClick={() => setChatOpen((v) => !v)}
          aria-label="Clinical AI Assistant Popup"
          title="Toggle Clinical AI Assistant"
        >
          <span className="chat-popup-dot" />
          <span className="chat-popup-label">{chatOpen ? 'Close Assistant' : 'AI Assistant'}</span>
          <svg className="chat-popup-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      </div>
    </>
  )
}
