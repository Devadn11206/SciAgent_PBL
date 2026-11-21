import { useEffect, useRef } from 'react'

interface Props {
  data: any[]
  layout?: any
  height?: number
  config?: any
  className?: string
}

export const FONT = { family: '"Plus Jakarta Sans", Inter, sans-serif', color: '#1e1b4b', size: 11 }

export function PlotlyChart({ data, layout, height = 320, config, className }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const plotly = useRef<any>(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      if (!plotly.current) plotly.current = (await import('plotly.js-dist-min')).default
      if (cancelled || !ref.current) return
      await plotly.current.react(
        ref.current,
        data,
        {
          autosize: true,
          height,
          margin: { l: 46, r: 14, t: 10, b: 40 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(255,255,255,0.0)',
          font: FONT,
          ...layout,
        },
        { displayModeBar: false, responsive: true, ...config },
      )
    })()
    return () => { cancelled = true }
  }, [data, layout, height, config])

  useEffect(() => {
    const el = ref.current
    return () => { if (el && plotly.current) plotly.current.purge(el) }
  }, [])

  return <div ref={ref} className={className} style={{ width: '100%', height }} />
}
