import type { ReactNode } from 'react'
import { useReveal } from '../useReveal'

interface Props {
  children: ReactNode
  index?: number
  className?: string
  as?: 'div' | 'section'
}

/** Wraps a block so it fades up on entry, staggered by `index`. */
export default function Reveal({ children, index = 0, className = '', as = 'div' }: Props) {
  const ref = useReveal<HTMLDivElement>()
  const Tag = as

  return (
    <Tag
      ref={ref}
      className={`reveal ${className}`.trim()}
      style={{ '--index': index } as React.CSSProperties}
    >
      {children}
    </Tag>
  )
}
