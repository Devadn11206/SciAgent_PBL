import { useState, useEffect } from 'react'
import { X, Check, Copy, Terminal, Database, Sparkles, Layers, Activity } from 'lucide-react'
import { fetchMCPManifest, type MCPManifestResponse } from '@/services/api'

interface Props {
  isOpen: boolean
  onClose: () => void
}

export function MCPInspectorModal({ isOpen, onClose }: Props) {
  const [manifest, setManifest] = useState<MCPManifestResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'tools' | 'resources' | 'prompts' | 'config'>('tools')
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setLoading(true)
      fetchMCPManifest()
        .then((m) => setManifest(m))
        .catch((e) => console.error('Failed to load MCP manifest:', e))
        .finally(() => setLoading(false))
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleCopyConfig = () => {
    if (!manifest?.claude_desktop_config) return
    navigator.clipboard.writeText(JSON.stringify(manifest.claude_desktop_config, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-indigo-950/40 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-3xl rounded-3xl bg-white shadow-2xl ring-1 ring-violet-200 overflow-hidden flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50/70 px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="flex h-3 w-3 rounded-full bg-emerald-500 animate-pulse" />
            <div>
              <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                FastMCP Core Inspector
                <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-semibold text-royal">
                  v2024-11-05
                </span>
              </h2>
              <p className="text-[11px] text-slate-500">
                Stateless Artifact Handles · Async Tasks · W3C Distributed Tracing
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-full p-1.5 text-slate-400 hover:bg-slate-200 hover:text-slate-600 transition"
          >
            <X size={18} />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-100 px-6 bg-white gap-4 text-xs font-semibold text-slate-600">
          <button
            onClick={() => setActiveTab('tools')}
            className={`py-3 flex items-center gap-1.5 border-b-2 transition ${
              activeTab === 'tools' ? 'border-royal text-royal' : 'border-transparent hover:text-slate-900'
            }`}
          >
            <Terminal size={14} /> Tools ({manifest?.tools.length || 5})
          </button>
          <button
            onClick={() => setActiveTab('resources')}
            className={`py-3 flex items-center gap-1.5 border-b-2 transition ${
              activeTab === 'resources' ? 'border-royal text-royal' : 'border-transparent hover:text-slate-900'
            }`}
          >
            <Database size={14} /> Resources ({manifest?.resources.length || 2})
          </button>
          <button
            onClick={() => setActiveTab('prompts')}
            className={`py-3 flex items-center gap-1.5 border-b-2 transition ${
              activeTab === 'prompts' ? 'border-royal text-royal' : 'border-transparent hover:text-slate-900'
            }`}
          >
            <Sparkles size={14} /> Prompts ({manifest?.prompts.length || 2})
          </button>
          <button
            onClick={() => setActiveTab('config')}
            className={`py-3 flex items-center gap-1.5 border-b-2 transition ${
              activeTab === 'config' ? 'border-royal text-royal' : 'border-transparent hover:text-slate-900'
            }`}
          >
            <Layers size={14} /> Client Config (Claude / Cursor)
          </button>
        </div>

        {/* Tab Body */}
        <div className="p-6 overflow-y-auto flex-1 bg-slate-50/50">
          {loading ? (
            <div className="flex h-48 items-center justify-center text-xs text-slate-400">
              Querying FastMCP server registry...
            </div>
          ) : activeTab === 'tools' ? (
            <div className="space-y-3">
              {manifest?.tools.map((t) => (
                <div key={t.name} className="rounded-2xl border border-violet-100 bg-white p-4 shadow-sm">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="mono font-bold text-xs text-royal">{t.name}()</span>
                    <span className="rounded bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
                      Callable Tool
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mb-2">{t.description}</p>
                </div>
              ))}
            </div>
          ) : activeTab === 'resources' ? (
            <div className="space-y-3">
              {manifest?.resources.map((r) => (
                <div key={r.uri} className="rounded-2xl border border-violet-100 bg-white p-4 shadow-sm">
                  <div className="mono font-bold text-xs text-royal mb-1">{r.uri}</div>
                  <p className="text-xs text-slate-600">{r.description || 'Live experiment state resource'}</p>
                </div>
              ))}
            </div>
          ) : activeTab === 'prompts' ? (
            <div className="space-y-3">
              {manifest?.prompts.map((p) => (
                <div key={p.name} className="rounded-2xl border border-violet-100 bg-white p-4 shadow-sm">
                  <div className="mono font-bold text-xs text-royal mb-1">{p.name}</div>
                  <p className="text-xs text-slate-600">{p.description || 'Pre-templated expert evaluation prompt'}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-700">
                  Add to <code className="mono text-royal">claude_desktop_config.json</code>:
                </span>
                <button
                  onClick={handleCopyConfig}
                  className="btn-ghost !py-1 text-xs flex items-center gap-1.5"
                >
                  {copied ? <Check size={13} className="text-emerald-600" /> : <Copy size={13} />}
                  {copied ? 'Copied JSON!' : 'Copy Config'}
                </button>
              </div>
              <pre className="rounded-2xl bg-slate-900 p-4 text-[12px] font-mono text-emerald-400 overflow-x-auto shadow-inner">
                {JSON.stringify(manifest?.claude_desktop_config || {}, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="border-t border-slate-100 bg-white px-6 py-3 flex items-center justify-between text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <Activity size={13} className="text-royal" />
            Stateless Artifacts: <code className="mono text-slate-700">artifact://run_id/type</code>
          </span>
          <button onClick={onClose} className="btn-primary !py-1.5 !px-4 text-xs">
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
