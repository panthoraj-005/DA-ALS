import { useEffect } from 'react'

interface Props {
  open: boolean
  onClose: () => void
}

interface Author {
  name: string
  handle: string
  role: string
}

const AUTHORS: Author[] = [
  {
    name: 'Panthoraj',
    handle: 'panthoraj-005',
    role: 'Signal pipeline, CNN training, and the screening backend.',
  },
  {
    name: 'PBS',
    handle: 'pbs002-s',
    role: 'Vision-language integration, interface, and deployment.',
  },
]

interface Dependency {
  name: string
  note: string
}

const BUILT_WITH: Dependency[] = [
  {
    name: 'Florence-2',
    note: 'Microsoft, MIT licence — reads the EMG trace as an image and describes what it sees.',
  },
  {
    name: 'PyTorch',
    note: 'Trains and serves the 1D convolutional network over the raw signal window.',
  },
  {
    name: 'XGBoost + SHAP',
    note: 'Gradient-boosted decision on engineered features, with per-feature attribution.',
  },
  {
    name: 'Google Gemini',
    note: 'Writes the plain-language explanation and answers follow-up questions.',
  },
  {
    name: 'FastAPI + React',
    note: 'Serves the inference API and renders this interface.',
  },
]

const REPO_URL = 'https://github.com/pbs002-s/medicalLLM'

export default function Credits({ open, onClose }: Props) {
  // Escape closes the dialog, matching the backdrop click and the header button.
  useEffect(() => {
    if (!open) return
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="credits-backdrop" onClick={onClose} role="presentation">
      <div
        className="credits-dialog panel"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="credits-title"
      >
        <div className="credits-head">
          <div>
            <p className="eyebrow">EMG ALS screening</p>
            <h2 className="credits-title" id="credits-title">
              Credits
            </h2>
          </div>
          <button
            type="button"
            className="button-quiet button-mini"
            onClick={onClose}
            aria-label="Close credits"
          >
            Close
          </button>
        </div>

        <div className="credits-body">
          <section className="credits-section">
            <p className="credits-label">Authors</p>
            <ul className="credits-authors">
              {AUTHORS.map((author) => (
                <li className="credits-author" key={author.handle}>
                  <p className="credits-author-name">{author.name}</p>
                  <p className="credits-author-role">{author.role}</p>
                  <a
                    className="credits-link num"
                    href={`https://github.com/${author.handle}`}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    @{author.handle}
                  </a>
                </li>
              ))}
            </ul>
          </section>

          <section className="credits-section">
            <p className="credits-label">Built with</p>
            <ul className="credits-stack">
              {BUILT_WITH.map((item) => (
                <li className="credits-stack-row" key={item.name}>
                  <span className="credits-stack-name num">{item.name}</span>
                  <span className="credits-stack-note">{item.note}</span>
                </li>
              ))}
            </ul>
          </section>

          <footer className="credits-footer">
            <p>
              Released under the MIT License ·{' '}
              <a
                className="credits-link"
                href={REPO_URL}
                target="_blank"
                rel="noreferrer noopener"
              >
                pbs002-s/medicalLLM
              </a>
            </p>
            <p>
              A research and screening aid, not a medical device. Nothing here is a clinical
              diagnosis.
            </p>
          </footer>
        </div>
      </div>
    </div>
  )
}
