import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'

export function Equation({ latex, display = true, className }: { latex: string; display?: boolean; className?: string }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: 'ignore' })
    } catch {
      return `<span style="color:#be185d">Invalid LaTeX</span>`
    }
  }, [latex, display])
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
