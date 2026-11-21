import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Check } from 'lucide-react'

export function cx(...a: (string | false | null | undefined)[]) {
  return a.filter(Boolean).join(' ')
}

export function Card({ title, icon, right, className, children }: { title?: string; icon?: ReactNode; right?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <section className={cx('glass p-5', className)}>
      {(title || right) && (
        <header className="mb-4 flex items-center justify-between gap-3">
          <h3 className="flex items-center gap-2 text-[15px] font-bold text-indigo-950">
            {icon}
            {title}
          </h3>
          {right}
        </header>
      )}
      {children}
    </section>
  )
}

export function Segmented<T extends string>({ value, onChange, options, small }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[]; small?: boolean }) {
  return (
    <div className="inline-flex rounded-full border border-violet-100 bg-violet-50/70 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            'rounded-full font-semibold transition',
            small ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-sm',
            value === o.value ? 'bg-royal text-white shadow' : 'text-violet-700 hover:bg-white/70',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: ReactNode }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)} className="inline-flex items-center gap-2 text-xs font-semibold text-indigo-900">
      <span className={cx('relative h-5 w-9 rounded-full transition', checked ? 'bg-royal' : 'bg-violet-200')}>
        <span className={cx('absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all', checked ? 'left-[18px]' : 'left-0.5')} />
      </span>
      {label}
    </button>
  )
}

export function Checkbox({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <button type="button" onClick={() => onChange(!checked)} className="inline-flex items-center gap-2 text-sm font-semibold text-indigo-900">
      <span className={cx('flex h-5 w-5 items-center justify-center rounded-md border transition', checked ? 'border-royal bg-royal text-white' : 'border-violet-200 bg-white')}>
        {checked && <Check size={13} strokeWidth={3} />}
      </span>
      {label}
    </button>
  )
}

export function Dropdown({ trigger, children, align = 'right' }: { trigger: (open: boolean, toggle: () => void) => ReactNode; children: (close: () => void) => ReactNode; align?: 'left' | 'right' }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])
  return (
    <div ref={ref} className="relative">
      {trigger(open, () => setOpen((o) => !o))}
      {open && (
        <div className={cx('absolute top-full z-50 mt-2 min-w-[300px] rounded-2xl border border-white bg-white/95 p-2 shadow-glass backdrop-blur-xl', align === 'right' ? 'right-0' : 'left-0')}>
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  )
}

export function Badge({ children, color = '#7C3AED', className }: { children: ReactNode; color?: string; className?: string }) {
  return (
    <span className={cx('inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-bold', className)} style={{ color, background: `${color}18`, border: `1px solid ${color}30` }}>
      {children}
    </span>
  )
}
