import { useCallback, useEffect, useRef, useState } from 'react'
import { assetUrl } from '../api'

/**
 * Plays the preprocessed EMG trace through the Web Audio API.
 *
 * The backend serves the same z-scored samples the plot shows, as raw
 * little-endian float32. We peak-normalise them to [-1, 1] and hand them to an
 * AudioBuffer.
 *
 * Rate, and why it is not just `sampleRate`: an AudioBuffer cannot be created at
 * the recording's 1 kHz — browsers only accept roughly 3 kHz to 384 kHz. So the
 * buffer is built at the context's own rate and `playbackRate` does the work:
 * `fs / ctx.sampleRate` consumes one stored sample per 1/fs second, which is
 * true time. The speed control multiplies that. At 1x the 5-450 Hz band lands
 * squarely in hearing range, so fibrillations crackle and fasciculations thump
 * the way they do over a needle EMG loudspeaker; 2x and 4x shorten a 23 s
 * recording for scanning, at the cost of transposing the pitch up. 8x matches
 * the `fs * 8` buffer rate sketched in docs/FUTURE_UPGRADES.md RFC-05.
 */

const SPEEDS = [0.5, 1, 2, 4, 8]
const DEFAULT_VOLUME = 0.8

type Status = 'idle' | 'loading' | 'ready' | 'error'

interface Props {
  url: string | null
  sampleRate: number
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const whole = Math.floor(seconds)
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`
}

export default function AudioPlayer({ url, sampleRate }: Props) {
  const contextRef = useRef<AudioContext | null>(null)
  const gainRef = useRef<GainNode | null>(null)
  const bufferRef = useRef<AudioBuffer | null>(null)
  const sourceRef = useRef<AudioBufferSourceNode | null>(null)
  // Position is tracked in *buffer* seconds so a speed change does not move it.
  const offsetRef = useRef(0)
  const startedAtRef = useRef(0)
  const frameRef = useRef(0)
  const stoppingRef = useRef(false)
  // The rate the live source is running at. Measuring elapsed time with the
  // newly selected rate instead would jump the position on every speed change.
  const rateRef = useRef(1)

  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')
  const [playing, setPlaying] = useState(false)
  const [volume, setVolume] = useState(DEFAULT_VOLUME)
  const [speed, setSpeed] = useState(1)
  const [progress, setProgress] = useState(0)
  const [bufferSeconds, setBufferSeconds] = useState(0)

  const src = assetUrl(url)
  // Real seconds of listening: buffer seconds slowed to the recording's own rate.
  const contextRate = contextRef.current?.sampleRate ?? 48_000
  const playbackRate = (sampleRate * speed) / contextRate
  const duration = playbackRate > 0 ? bufferSeconds / playbackRate : 0

  const stopSource = useCallback(() => {
    const source = sourceRef.current
    if (!source) return
    stoppingRef.current = true
    try {
      source.stop()
    } catch {
      /* already stopped — nothing to undo */
    }
    source.disconnect()
    sourceRef.current = null
    stoppingRef.current = false
  }, [])

  /** Current position in buffer seconds, including the part played since start. */
  const currentOffset = useCallback(() => {
    const ctx = contextRef.current
    if (!ctx || !sourceRef.current) return offsetRef.current
    const played = (ctx.currentTime - startedAtRef.current) * rateRef.current
    return Math.min(offsetRef.current + played, bufferRef.current?.duration ?? 0)
  }, [])

  const tick = useCallback(() => {
    const total = bufferRef.current?.duration ?? 0
    if (total > 0) setProgress(currentOffset() / total)
    frameRef.current = requestAnimationFrame(tick)
  }, [currentOffset])

  /** Start playing from `offset` buffer seconds. The caller sets offsetRef. */
  const startFrom = useCallback(
    (offset: number) => {
      const ctx = contextRef.current
      const buffer = bufferRef.current
      const gain = gainRef.current
      if (!ctx || !buffer || !gain) return

      stopSource()

      const source = ctx.createBufferSource()
      source.buffer = buffer
      rateRef.current = (sampleRate * speed) / ctx.sampleRate
      source.playbackRate.value = rateRef.current
      source.connect(gain)
      source.onended = () => {
        if (stoppingRef.current) return // a pause or a seek, not the end
        offsetRef.current = 0
        setProgress(0)
        setPlaying(false)
        sourceRef.current = null
      }

      offsetRef.current = offset
      startedAtRef.current = ctx.currentTime
      source.start(0, offset)
      sourceRef.current = source
      setPlaying(true)
    },
    [sampleRate, speed, stopSource],
  )

  /** Fetch the samples once, on the first play — this is ~90 KB per recording. */
  const load = useCallback(async (): Promise<boolean> => {
    if (bufferRef.current) return true
    if (!src) return false

    setStatus('loading')
    setError('')
    try {
      const response = await fetch(src)
      if (!response.ok) throw new Error(`server returned ${response.status}`)

      const samples = new Float32Array(await response.arrayBuffer())
      if (samples.length === 0) throw new Error('the recording held no samples')

      const ctx = contextRef.current ?? new AudioContext()
      contextRef.current = ctx
      if (!gainRef.current) {
        const gain = ctx.createGain()
        gain.gain.value = volume
        gain.connect(ctx.destination)
        gainRef.current = gain
      }

      // Peak-normalise to [-1, 1]; a z-scored trace is not bounded on its own.
      let peak = 0
      for (const value of samples) {
        const magnitude = Math.abs(value)
        if (magnitude > peak) peak = magnitude
      }
      const scale = peak > 0 ? 1 / peak : 0

      const buffer = ctx.createBuffer(1, samples.length, ctx.sampleRate)
      const channel = buffer.getChannelData(0)
      for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] * scale

      bufferRef.current = buffer
      setBufferSeconds(buffer.duration)
      setStatus('ready')
      return true
    } catch (cause) {
      setStatus('error')
      setError(
        cause instanceof Error && cause.message
          ? `Audio unavailable: ${cause.message}. Generated artifacts expire — re-run the signal.`
          : 'Audio unavailable. Re-run the signal to rebuild it.',
      )
      return false
    }
  }, [src, volume])

  async function toggle() {
    if (playing) {
      offsetRef.current = currentOffset()
      stopSource()
      setPlaying(false)
      return
    }

    if (!(await load())) return
    // Browsers start contexts suspended until a gesture; this click is one.
    if (contextRef.current?.state === 'suspended') await contextRef.current.resume()
    startFrom(offsetRef.current)
  }

  function seek(event: React.MouseEvent<HTMLDivElement>) {
    const total = bufferRef.current?.duration ?? 0
    if (total === 0) return

    const box = event.currentTarget.getBoundingClientRect()
    const fraction = Math.min(Math.max((event.clientX - box.left) / box.width, 0), 1)
    const offset = fraction * total

    setProgress(fraction)
    if (playing) startFrom(offset)
    else offsetRef.current = offset
  }

  // Live volume, without restarting playback.
  useEffect(() => {
    if (gainRef.current) gainRef.current.gain.value = volume
  }, [volume])

  // Speed changes apply to the running source in place, so the position holds:
  // bank the offset at the old rate, then switch the rate and restart the clock.
  useEffect(() => {
    const ctx = contextRef.current
    const source = sourceRef.current
    if (!ctx || !source) return
    offsetRef.current = currentOffset()
    startedAtRef.current = ctx.currentTime
    rateRef.current = (sampleRate * speed) / ctx.sampleRate
    source.playbackRate.value = rateRef.current
  }, [speed, sampleRate, currentOffset])

  // Only animate while something is actually playing.
  useEffect(() => {
    if (!playing) return
    frameRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frameRef.current)
  }, [playing, tick])

  // A new prediction is a new recording: drop the decoded buffer and rewind.
  useEffect(() => {
    stopSource()
    bufferRef.current = null
    offsetRef.current = 0
    setPlaying(false)
    setProgress(0)
    setBufferSeconds(0)
    setStatus('idle')
    setError('')
  }, [src, stopSource])

  // An AudioContext is a real audio device handle; close it with the component.
  useEffect(() => {
    return () => {
      stopSource()
      void contextRef.current?.close()
      contextRef.current = null
      gainRef.current = null
    }
  }, [stopSource])

  if (!src) return null

  const elapsed = duration * progress

  return (
    <div className="audio-player">
      <button
        type="button"
        className="audio-play"
        onClick={() => void toggle()}
        disabled={status === 'loading'}
        aria-label={playing ? 'Pause the EMG audio' : 'Listen to the EMG audio'}
      >
        {status === 'loading' ? (
          <span className="audio-spinner" aria-hidden="true" />
        ) : playing ? (
          <svg width="12" height="13" viewBox="0 0 12 13" aria-hidden="true" fill="currentColor">
            <rect x="1" y="1" width="3.5" height="11" rx="1" />
            <rect x="7.5" y="1" width="3.5" height="11" rx="1" />
          </svg>
        ) : (
          <svg width="12" height="13" viewBox="0 0 12 13" aria-hidden="true" fill="currentColor">
            <path d="M2 1.5v10l9-5z" />
          </svg>
        )}
        <span>{playing ? 'Pause' : 'Listen to EMG'}</span>
      </button>

      <div
        className="audio-track"
        onClick={seek}
        role="presentation"
        title="Click to scrub through the recording"
      >
        <div className="audio-track-fill" style={{ width: `${progress * 100}%` }} />
      </div>

      <span className="num audio-time">
        {formatTime(elapsed)} / {formatTime(duration)}
      </span>

      <label className="audio-control">
        <span className="eyebrow">Vol</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(event) => setVolume(Number(event.target.value))}
          aria-label="Playback volume"
        />
      </label>

      <div className="audio-speeds" role="group" aria-label="Playback speed">
        {SPEEDS.map((option) => (
          <button
            key={option}
            type="button"
            className={`audio-speed${option === speed ? ' is-active' : ''}`}
            onClick={() => setSpeed(option)}
            aria-pressed={option === speed}
          >
            {option}x
          </button>
        ))}
      </div>

      {status === 'error' && <p className="audio-error">{error}</p>}
    </div>
  )
}
