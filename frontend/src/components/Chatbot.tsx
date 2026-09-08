import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { sendChatMessage } from '../api'
import type { ChatMessage, PredictionResult } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  currentResult: PredictionResult | null
  onOpenSettings?: () => void
}

const TEXTAREA_MAX_PX = 120

function getContextualStarters(result: PredictionResult | null): string[] {
  if (!result) {
    return [
      'How do 1-D CNN and Florence-2 analyze EMG signals differently?',
      'What is the neurophysiological significance of abnormal segments?',
      'How does the XGBoost meta-learner compute its confidence score?',
      'What are the primary electromyographic signs of ALS denervation?',
    ]
  }

  const isAls = result.final_prediction === 'ALS'
  const confidencePct = (result.final_confidence * 100).toFixed(1)
  const abnormal = result.abnormal_segments || '2/10'

  if (isAls) {
    return [
      `Why did the models classify Signal ${result.id} as ALS (${confidencePct}%)?`,
      `What pathological features were detected in the ${abnormal} abnormal segments?`,
      `How do the CNN (${((result.cnn_probability ?? 0.9) * 100).toFixed(0)}%) and Florence-2 (${((result.florence_probability ?? 0.9) * 100).toFixed(0)}%) scores compare?`,
      `What role did ${result.top_shap_feature || 'CNN_Prob'} play in this screening verdict?`,
    ]
  }

  return [
    `Why was Signal ${result.id} categorized as Normal (${confidencePct}%)?`,
    `What waveform characteristics ruled out active motor unit denervation?`,
    `How do the baseline segment energies in this recording compare to ALS traces?`,
    `How did the meta-learner evaluate the agreement between CNN and Florence-2?`,
  ]
}

/* --- lightweight prose rendering ------------------------------------------
   The model answers in light markdown. Rather than pull in a parser — or hand
   the reply to dangerouslySetInnerHTML — map the handful of shapes it actually
   emits (headings, bullets, numbered steps, bold, inline code) onto real
   elements. Anything unrecognised stays literal text.
   -------------------------------------------------------------------------- */

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*)/g

function renderInline(text: string, key: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = []
  let cursor = 0
  let index = 0
  let match: RegExpExecArray | null

  INLINE.lastIndex = 0
  while ((match = INLINE.exec(text)) !== null) {
    if (match.index > cursor) nodes.push(text.slice(cursor, match.index))
    const token = match[0]
    if (token.startsWith('**')) {
      nodes.push(<strong key={`${key}-b${index}`}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('`')) {
      nodes.push(
        <code className="chat-code" key={`${key}-c${index}`}>
          {token.slice(1, -1)}
        </code>,
      )
    } else {
      nodes.push(<em key={`${key}-i${index}`}>{token.slice(1, -1)}</em>)
    }
    cursor = match.index + token.length
    index += 1
  }
  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

function renderRich(text: string): React.ReactNode[] {
  const out: React.ReactNode[] = []
  let paragraph: string[] = []
  let listItems: string[] = []
  let listOrdered = false

  function flushParagraph() {
    if (paragraph.length === 0) return
    const key = `p${out.length}`
    out.push(<p key={key}>{renderInline(paragraph.join(' '), key)}</p>)
    paragraph = []
  }

  function flushList() {
    if (listItems.length === 0) return
    const key = `l${out.length}`
    const items = listItems.map((item, i) => <li key={i}>{renderInline(item, `${key}-${i}`)}</li>)
    out.push(listOrdered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>)
    listItems = []
  }

  for (const raw of text.split('\n')) {
    const line = raw.trim()

    if (line === '') {
      flushParagraph()
      flushList()
      continue
    }

    const heading = /^#{2,4}\s+(.*)$/.exec(line)
    if (heading) {
      flushParagraph()
      flushList()
      const key = `h${out.length}`
      out.push(<h4 key={key}>{renderInline(heading[1], key)}</h4>)
      continue
    }

    const bullet = /^[-*•]\s+(.*)$/.exec(line)
    if (bullet) {
      flushParagraph()
      if (listItems.length > 0 && listOrdered) flushList()
      listOrdered = false
      listItems.push(bullet[1])
      continue
    }

    const numbered = /^\d+[.)]\s+(.*)$/.exec(line)
    if (numbered) {
      flushParagraph()
      if (listItems.length > 0 && !listOrdered) flushList()
      listOrdered = true
      listItems.push(numbered[1])
      continue
    }

    flushList()
    paragraph.push(line)
  }

  flushParagraph()
  flushList()
  return out
}

export default function Chatbot({ open, onClose, currentResult, onOpenSettings }: Props) {
  const starters = useMemo(() => getContextualStarters(currentResult), [currentResult])

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content:
        'Clinical AI Assistant online. Specialized in EMG waveform interpretation, ALS neurophysiology, and screening model outputs (CNN, Florence-2 VLM, XGBoost).',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [startersOpen, setStartersOpen] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  const hasConversation = messages.some((m) => m.role === 'user')

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 120)
    }
  }, [open])

  // Escape closes the drawer, matching the backdrop click and the header button.
  useEffect(() => {
    if (!open) return
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  // Drop any in-flight request when the drawer closes, and on unmount.
  useEffect(() => {
    if (!open) abortRef.current?.abort()
  }, [open])

  useEffect(() => () => abortRef.current?.abort(), [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  function growTextarea(el: HTMLTextAreaElement) {
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, TEXTAREA_MAX_PX)}px`
  }

  const handleSend = useCallback(
    async function send(textToSend?: string) {
      const query = (textToSend ?? input).trim()
      if (!query || loading) return

      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: query,
        timestamp: new Date(),
      }

      let history: { role: string; content: string }[] = []
      setMessages((prev) => {
        const next = [...prev, userMsg]
        history = next.slice(-6).map((m) => ({ role: m.role, content: m.content }))
        return next
      })
      setInput('')
      setStartersOpen(false)
      if (inputRef.current) inputRef.current.style.height = 'auto'
      setLoading(true)

      const controller = new AbortController()
      abortRef.current = controller

      try {
        const contextPayload = currentResult
          ? {
              id: currentResult.id,
              source: currentResult.source,
              final_prediction: currentResult.final_prediction,
              final_confidence: currentResult.final_confidence,
              severity: currentResult.severity,
              cnn_probability: currentResult.cnn_probability,
              florence_probability: currentResult.florence_probability,
              models_agree: currentResult.models_agree,
              abnormal_segments: currentResult.abnormal_segments,
              top_shap_feature: currentResult.top_shap_feature,
              explanation: currentResult.explanation,
            }
          : undefined

        const { reply, suggestions } = await sendChatMessage(
          query,
          history,
          contextPayload,
          controller.signal,
        )

        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: reply,
          suggestions: suggestions && suggestions.length > 0 ? suggestions : undefined,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, assistantMsg])
      } catch (err) {
        // A user-pressed Stop is not a failure — say so instead of raising an error.
        const cancelled = controller.signal.aborted
        const errorMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: cancelled
            ? 'Generation stopped.'
            : err instanceof Error
              ? `Error: ${err.message}`
              : 'Communication failed. Please check network connectivity.',
          failedQuery: cancelled ? undefined : query,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, errorMsg])
      } finally {
        abortRef.current = null
        setLoading(false)
      }
    },
    [currentResult, input, loading],
  )

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
  }

  function copyMessage(id: string, text: string) {
    void navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 1800)
  }

  function clearHistory() {
    abortRef.current?.abort()
    setMessages([
      {
        id: `init-${Date.now()}`,
        role: 'assistant',
        content: 'Conversation history reset. How can I assist with this EMG screening?',
        timestamp: new Date(),
      },
    ])
    setStartersOpen(true)
  }

  const placeholderText = currentResult
    ? `Ask about Signal ${currentResult.id} (${currentResult.final_prediction} · ${(currentResult.final_confidence * 100).toFixed(0)}%)...`
    : 'Ask a clinical question on EMG signals or AI models...'

  if (!open) return null

  const showStarters = startersOpen || !hasConversation

  return (
    <div className="chat-backdrop" onClick={onClose} role="presentation">
      <div
        className="chat-drawer"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Clinical AI Assistant"
      >
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-info">
            <span className={`chat-avatar-indicator ${loading ? 'is-thinking' : ''}`} />
            <div>
              <h3 className="chat-title">Clinical AI Assistant</h3>
              <p className="eyebrow" style={{ marginTop: '2px' }}>
                ALS Neurophysiology &amp; Models
              </p>
            </div>
          </div>
          <div className="chat-header-actions">
            {onOpenSettings && (
              <button
                type="button"
                className="button-quiet button-mini"
                onClick={onOpenSettings}
                title="Configure AI Provider & Model"
              >
                Settings
              </button>
            )}
            <button
              type="button"
              className="button-quiet button-mini"
              onClick={clearHistory}
              disabled={!hasConversation && !loading}
            >
              Clear
            </button>
            <button
              type="button"
              className="button-quiet button-mini"
              onClick={onClose}
              aria-label="Close assistant"
            >
              <svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M1 1L13 13M13 1L1 13" />
              </svg>
            </button>
          </div>
        </div>

        {/* Current screening context banner */}
        {currentResult ? (
          <div className="chat-context-bar">
            <span className="eyebrow">Active Context:</span>
            <span className="num">Signal {currentResult.id}</span>
            <span className={`tag ${currentResult.final_prediction === 'ALS' ? 'is-red' : 'is-green'}`}>
              <span className="tag-dot" />
              {currentResult.final_prediction} ({(currentResult.final_confidence * 100).toFixed(1)}%)
            </span>
          </div>
        ) : (
          <div className="chat-context-bar is-neutral">
            <span className="eyebrow">Context:</span>
            <span className="chat-context-hint">No active signal selected. General questions mode.</span>
          </div>
        )}

        {/* Messages List */}
        <div className="chat-messages" aria-live="polite" aria-busy={loading}>
          {messages.map((m) => (
            <div key={m.id} className={`chat-message ${m.role === 'user' ? 'is-user' : 'is-assistant'}`}>
              <div className="chat-message-meta">
                <span className="eyebrow">{m.role === 'user' ? 'Clinician' : 'AI Assistant'}</span>
                <span className="num chat-time">
                  {m.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className={`chat-message-bubble ${m.failedQuery ? 'is-failed' : ''}`}>
                <div className="chat-message-text">
                  {m.role === 'assistant' ? renderRich(m.content) : m.content}
                </div>

                {/* Dynamic Follow-up Questions Generated by Assistant */}
                {m.suggestions && m.suggestions.length > 0 && (
                  <div className="chat-related-box">
                    <div className="chat-related-head">
                      <span className="eyebrow">Related Inquiries</span>
                    </div>
                    <div className="chat-related-list">
                      {m.suggestions.map((sug, sIdx) => (
                        <button
                          key={sIdx}
                          type="button"
                          className="chat-related-chip"
                          disabled={loading}
                          onClick={() => void handleSend(sug)}
                        >
                          <span className="chat-related-bullet">↳</span>
                          <span className="chat-related-text">{sug}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {m.role === 'assistant' && (
                  <div className="chat-bubble-actions">
                    {onOpenSettings && (m.content.includes('Settings') || m.content.includes('API key')) && (
                      <button
                        type="button"
                        className="chat-copy-btn"
                        onClick={onOpenSettings}
                        style={{ color: 'var(--accent, #2563eb)', fontWeight: 600 }}
                      >
                        ⚙ Configure in Settings
                      </button>
                    )}
                    {m.failedQuery && (
                      <button
                        type="button"
                        className="chat-copy-btn"
                        disabled={loading}
                        onClick={() => void handleSend(m.failedQuery)}
                      >
                        Retry
                      </button>
                    )}
                    <button
                      type="button"
                      className="chat-copy-btn"
                      onClick={() => copyMessage(m.id, m.content)}
                    >
                      {copiedId === m.id ? 'Copied' : 'Copy'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="chat-message is-assistant">
              <div className="chat-message-meta">
                <span className="eyebrow">AI Assistant</span>
              </div>
              <div className="chat-message-bubble chat-loading-bubble">
                <div className="chat-typing-dots">
                  <span />
                  <span />
                  <span />
                </div>
                <span className="chat-loading-text">Analyzing query...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Queries — Minimal Vertical Alignment */}
        <div className="chat-starters">
          <div className="chat-starters-head">
            <span className="eyebrow">Suggested Inquiries</span>
            {hasConversation && (
              <button
                type="button"
                className="chat-copy-btn"
                aria-expanded={showStarters}
                onClick={() => setStartersOpen((v) => !v)}
              >
                {showStarters ? 'Hide' : 'Show'}
              </button>
            )}
          </div>
          {showStarters && (
            <div className="chat-starters-list">
              {starters.map((q, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="chat-starter-row"
                  disabled={loading}
                  onClick={() => void handleSend(q)}
                  title={q}
                >
                  <span className="num chat-starter-num">0{idx + 1}</span>
                  <span className="chat-starter-text">{q}</span>
                  <span className="chat-starter-arrow">→</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="chat-input-area">
          <div className="chat-input-wrap">
            <textarea
              ref={inputRef}
              className="chat-textarea"
              placeholder={placeholderText}
              value={input}
              rows={1}
              disabled={loading}
              onChange={(e) => {
                setInput(e.target.value)
                growTextarea(e.target)
              }}
              onKeyDown={handleKeyDown}
            />
            <span className="chat-input-hint num">Enter sends · Shift+Enter newline</span>
          </div>
          {loading ? (
            <button
              type="button"
              className="chat-send-btn is-stop"
              onClick={() => abortRef.current?.abort()}
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="chat-send-btn"
              disabled={!input.trim()}
              onClick={() => void handleSend()}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
