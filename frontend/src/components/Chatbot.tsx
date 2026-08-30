import { useEffect, useMemo, useRef, useState } from 'react'
import { sendChatMessage } from '../api'
import type { ChatMessage, PredictionResult } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  currentResult: PredictionResult | null
}

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

export default function Chatbot({ open, onClose, currentResult }: Props) {
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
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 120)
    }
  }, [open])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function handleSend(textToSend?: string) {
    const query = (textToSend ?? input).trim()
    if (!query || loading) return

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: query,
      timestamp: new Date(),
    }

    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    setInput('')
    setLoading(true)

    try {
      const historyPayload = nextMessages
        .slice(-6)
        .map((m) => ({ role: m.role, content: m.content }))

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

      const { reply, suggestions } = await sendChatMessage(query, historyPayload, contextPayload)

      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: reply,
        suggestions: suggestions && suggestions.length > 0 ? suggestions : undefined,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      const errorMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content:
          err instanceof Error
            ? `Error: ${err.message}`
            : 'Communication failed. Please check network connectivity.',
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setLoading(false)
    }
  }

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
    setMessages([
      {
        id: `init-${Date.now()}`,
        role: 'assistant',
        content: 'Conversation history reset. How can I assist with this EMG screening?',
        timestamp: new Date(),
      },
    ])
  }

  const placeholderText = currentResult
    ? `Ask about Signal ${currentResult.id} (${currentResult.final_prediction} · ${(currentResult.final_confidence * 100).toFixed(0)}%)...`
    : 'Ask a clinical question on EMG signals or AI models...'

  if (!open) return null

  return (
    <div className="chat-backdrop" onClick={onClose} role="presentation">
      <div
        className="chat-drawer"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-label="Clinical AI Assistant"
      >
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-info">
            <span className="chat-avatar-indicator" />
            <div>
              <h3 className="chat-title">Clinical AI Assistant</h3>
              <p className="eyebrow" style={{ marginTop: '2px' }}>
                ALS Neurophysiology & Models
              </p>
            </div>
          </div>
          <div className="chat-header-actions">
            <button
              type="button"
              className="button-quiet button-mini"
              onClick={clearHistory}
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
                <path d="M1 1L13 13M1 13L1" />
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
        <div className="chat-messages">
          {messages.map((m) => (
            <div key={m.id} className={`chat-message ${m.role === 'user' ? 'is-user' : 'is-assistant'}`}>
              <div className="chat-message-meta">
                <span className="eyebrow">{m.role === 'user' ? 'Clinician' : 'AI Assistant'}</span>
                <span className="num chat-time">
                  {m.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </div>
              <div className="chat-message-bubble">
                <div className="chat-message-text">{m.content}</div>

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
          </div>
          <div className="chat-starters-list">
            {starters.map((q, idx) => (
              <button
                key={idx}
                type="button"
                className="chat-starter-row"
                disabled={loading}
                onClick={() => void handleSend(q)}
              >
                <span className="num chat-starter-num">0{idx + 1}</span>
                <span className="chat-starter-text">{q}</span>
                <span className="chat-starter-arrow">→</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <div className="chat-input-area">
          <textarea
            ref={inputRef}
            className="chat-textarea"
            placeholder={placeholderText}
            value={input}
            rows={1}
            disabled={loading}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            type="button"
            className="chat-send-btn"
            disabled={!input.trim() || loading}
            onClick={() => void handleSend()}
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  )
}
