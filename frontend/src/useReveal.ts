import { useEffect, useRef } from 'react'

/**
 * Entry animation. IntersectionObserver only — never a scroll listener, which
 * would reflow on every frame. Elements that are already in view when they
 * mount still animate, so a result appearing below the fold and a result
 * appearing on screen behave the same.
 */
export function useReveal<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null)

  useEffect(() => {
    const node = ref.current
    if (!node) return

    if (
      typeof IntersectionObserver === 'undefined' ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      node.classList.add('is-in')
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-in')
            observer.unobserve(entry.target)
          }
        }
      },
      { threshold: 0.06, rootMargin: '0px 0px -40px 0px' },
    )

    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  return ref
}
