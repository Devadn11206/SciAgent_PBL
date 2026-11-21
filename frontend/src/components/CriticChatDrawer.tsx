import { useState } from 'react'
import { Bot, HelpCircle, Loader2, MessageSquare, Send, Sparkles, User } from 'lucide-react'
import { askCriticChat } from '@/services/api'
import { Badge, Card, cx } from '@/components/ui'

interface Props {
  runId: string | null
  problemName?: string
  className?: string
}

const QUICK_CHIPS = [
  'Why did Trial 1 struggle?',
  'How reliable is the discovered parameter?',
  'Summarize physical insights for my paper',
]

interface Message {
  role: 'user' | 'assistant'
  content: string
  model?: string
}

export function CriticChatDrawer({ runId, problemName, className }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(true)

  const handleSend = async (textToSend?: string) => {
    const q = (textToSend || input).trim()
    if (!q || !runId) return

    const newMsgs: Message[] = [...messages, { role: 'user', content: q }]
    setMessages(newMsgs)
    setInput('')
    setLoading(true)

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }))
      const res = await askCriticChat(runId, q, history)
      setMessages([...newMsgs, { role: 'assistant', content: res.reply, model: res.model }])
    } catch (err: any) {
      setMessages([
        ...newMsgs,
        {
          role: 'assistant',
          content: `Error contacting Critic Agent: ${err.message || 'Service unavailable'}`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className={cx('border-violet-200 bg-white/95 shadow-md', className)}>
      <div
        className="flex cursor-pointer items-center justify-between pb-2 border-b border-violet-100"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center gap-2">
          <div className="rounded-xl bg-violet-100 p-1.5 text-violet-700">
            <Bot size={18} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-indigo-950 flex items-center gap-2">
              Ask the Critic about this Run
              {runId ? (
                <Badge color="#7C3AED" className="text-[10px]">
                  {runId}
                </Badge>
              ) : (
                <Badge color="#94A3B8" className="text-[10px]">
                  Awaiting Active Run
                </Badge>
              )}
            </h3>
            <p className="text-[11px] text-slate-500">
              Grounded Ollama critic assistant querying live SQLite metrics & ablation logs
            </p>
          </div>
        </div>
        <button
          className="text-xs font-semibold text-violet-600 hover:text-violet-800"
          onClick={(e) => {
            e.stopPropagation()
            setIsOpen(!isOpen)
          }}
        >
          {isOpen ? 'Minimize' : 'Expand'}
        </button>
      </div>

      {isOpen && (
        <div className="mt-3 space-y-3">
          {/* Quick chips */}
          <div className="flex flex-wrap gap-1.5">
            {QUICK_CHIPS.map((chip) => (
              <button
                key={chip}
                disabled={!runId || loading}
                onClick={() => handleSend(chip)}
                className="flex items-center gap-1 rounded-full border border-violet-200 bg-violet-50/70 px-2.5 py-1 text-[11px] font-medium text-violet-800 transition hover:bg-violet-100 disabled:opacity-50"
              >
                <Sparkles size={11} className="text-violet-500" />
                {chip}
              </button>
            ))}
          </div>

          {/* Messages list */}
          <div className="max-h-60 min-h-24 space-y-2 overflow-y-auto rounded-xl border border-violet-100 bg-slate-50/60 p-3 text-xs">
            {messages.length === 0 ? (
              <div className="flex h-24 flex-col items-center justify-center text-center text-slate-400">
                <HelpCircle size={20} className="mb-1 text-violet-400/50" />
                <span>
                  {runId
                    ? 'Ask the Critic Agent about convergence, ablation diagnostics, or parameter reliability.'
                    : 'Execute or select an experiment to chat with the Critic.'}
                </span>
              </div>
            ) : (
              messages.map((m, idx) => (
                <div
                  key={idx}
                  className={cx(
                    'flex gap-2 rounded-xl p-2.5',
                    m.role === 'user' ? 'bg-violet-100/70 ml-8 text-indigo-950' : 'bg-white border border-violet-100 mr-8 text-slate-800 shadow-sm'
                  )}
                >
                  <div className="shrink-0 mt-0.5">
                    {m.role === 'user' ? (
                      <User size={14} className="text-violet-700" />
                    ) : (
                      <Bot size={14} className="text-rose-500" />
                    )}
                  </div>
                  <div className="flex-1 whitespace-pre-wrap leading-relaxed text-[12px]">
                    <div className="mb-0.5 font-bold text-[10px] text-slate-400 uppercase tracking-wider">
                      {m.role === 'user' ? 'You' : `Critic Agent ${m.model ? `(${m.model})` : ''}`}
                    </div>
                    {m.content}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="flex items-center gap-2 text-violet-600 p-2 text-xs italic">
                <Loader2 size={13} className="animate-spin" />
                Critic analyzing SQLite logs & formulating insight...
              </div>
            )}
          </div>

          {/* Input box */}
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={input}
              disabled={!runId || loading}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder={runId ? "Ask a question about this experiment..." : "Run or select an experiment first..."}
              className="flex-1 rounded-xl border border-violet-200 bg-white px-3.5 py-2 text-xs text-slate-800 placeholder-slate-400 shadow-sm focus:border-royal focus:outline-none"
            />
            <button
              disabled={!input.trim() || !runId || loading}
              onClick={() => handleSend()}
              className="flex items-center justify-center rounded-xl bg-royal px-3.5 py-2 text-xs font-bold text-white shadow transition hover:bg-violet-700 disabled:opacity-50"
            >
              <Send size={13} />
            </button>
          </div>
        </div>
      )}
    </Card>
  )
}
