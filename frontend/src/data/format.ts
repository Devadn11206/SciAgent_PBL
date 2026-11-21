export const sci = (v: number | null, digits = 2): string => {
  if (v === null || v === undefined) return '—'
  if (v === 0) return '0'
  const [m, e] = v.toExponential(digits).split('e')
  return `${m}e${parseInt(e, 10)}`
}
export const pct = (v: number, d = 2) => `${v.toFixed(d)}%`
export const latexSci = (v: number | null, digits = 2): string => {
  if (v === null) return '---'
  const [m, e] = v.toExponential(digits).split('e')
  return `$${m}\\times10^{${parseInt(e, 10)}}$`
}
