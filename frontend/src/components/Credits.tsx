import { useEffect } from 'react'

const REPO_URL = 'https://github.com/pbs002-s/medicalLLM'

interface Author {
  name: string
  handle: string
  role: string
}

const AUTHORS: Author[] = [
  {
    name: 'Panthoraj',
    handle: 'panthoraj-005',
    role: 'Multi-modal model architecture and training pipeline',
  },
  {
    name: 'PBS',
    handle: 'pbs002-s',
    role: 'Inference backend, explainability, and clinical interface',
  },
]

/** Third-party work the platform is built on. Kept short — the README carries the long form. */
const STACK: { label: string; detail: string }[] = [
  { label: 'Florence-2', detail: 'Vision-language backbone, Microsoft (MIT)' },
  { label: 'PyTorch', detail: '1-D CNN training and inference' },
  { label: 'XGBoost · SHAP', detail: 'Meta-learner and feature attribution' },
  { label: 'Google Gemini', detail: 'Clinical assistant language model' },
  { label: 'FastAPI · React', detail: 'Service layer and single-page interface' },
]

interface Props {
  open: boolean
  onClose: () => void
}

export default function Credits({ open, onClose }: Props) {
  // Escape closes the dialog, matching how the assistant drawer behaves.
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
        className="credits-modal"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="credits-title"
      >
        <div className="credits-head">
          <div>
            <p className="eyebrow">Credits</p>
            <h2 className="credits-title" id="credits-title">
              Built by
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
          <ul className="credits-authors">
            {AUTHORS.map((author) => (
              <li key={author.handle} className="credits-author">
                <div className="credits-author-top">
                  <span className="credits-author-name">{author.name}</span>
                  <a
                    className="credits-handle"
                    href={`https://github.com/${author.handle}`}
                    target="_blank"
                    rel="noreferrer noopener"
                  >
                    @{author.handle}
                  </a>
                </div>
                <p className="credits-author-role">{author.role}</p>
              </li>
            ))}
          </ul>

          <div className="credits-section">
            <p className="credits-section-head">Built with</p>
            <ul className="credits-stack">
              {STACK.map((item) => (
                <li key={item.label} className="credits-stack-row">
                  <span className="credits-stack-label">{item.label}</span>
                  <span className="credits-stack-detail">{item.detail}</span>
                </li>
              ))}
            </ul>
          </div>

          <p className="credits-foot">
            Released under the MIT License. Source at{' '}
            <a href={REPO_URL} target="_blank" rel="noreferrer noopener">
              pbs002-s/medicalLLM
            </a>
            . Research and screening aid only — not a medical device.
          </p>
        </div>
      </div>
    </div>
  )
}
