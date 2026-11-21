/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: { royal: '#7C3AED', turq: '#06B6D4', rosepink: '#EC4899', emer: '#10B981', indigoDeep: '#1E1B4B' },
      fontFamily: { sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'], mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'] },
      boxShadow: { glass: '0 8px 32px -8px rgba(124,58,237,0.18), 0 2px 8px -2px rgba(236,72,153,0.10)' },
    },
  },
  plugins: [],
}
