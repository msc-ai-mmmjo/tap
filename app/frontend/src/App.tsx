import { useRef, useEffect, useState } from 'react';
import { useChat } from './hooks/useChat';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';
import { ResponseSkeleton } from './components/ResponseSkeleton';

const SAMPLE_QUERIES = [
  "I'm admitting a patient in diabetic ketoacidosis, where do I start?",
  "My patient's been on a proton pump inhibitor for years, what long-term risks should I flag?",
  'Starting empirical antibiotics for community-acquired pneumonia in an otherwise healthy adult, which is first-line? A) Amoxicillin B) Azithromycin C) Levofloxacin D) Doxycycline',
  'Acute monoarticular knee pain, synovial fluid shows negatively birefringent needle-shaped crystals. Which fits? A) Gout B) Pseudogout C) Septic arthritis D) Rheumatoid arthritis',
];

function ExchangeSeparator({ turn }: { turn: number }) {
  return (
    <div className="my-8 flex items-center gap-3" aria-hidden>
      <div className="flex-1" style={{ borderTop: '1px solid var(--color-rule)' }} />
      <div
        className="font-mono text-[10px] uppercase tracking-[0.18em]"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        {String(turn).padStart(2, '0')}
      </div>
      <div className="flex-1" style={{ borderTop: '1px solid var(--color-rule)' }} />
    </div>
  );
}

function App() {
  const { messages, loading, error, send } = useChat();
  const lastMsgRef = useRef<HTMLDivElement>(null);
  const [showSkeleton, setShowSkeleton] = useState(false);

  useEffect(() => {
    lastMsgRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [messages]);

  useEffect(() => {
    if (!loading) {
      setShowSkeleton(false);
      return;
    }
    const id = setTimeout(() => setShowSkeleton(true), 200);
    return () => clearTimeout(id);
  }, [loading]);

  return (
    <div className="h-screen flex flex-col" style={{ background: 'var(--color-paper)' }}>
      <header
        className="shrink-0"
        style={{ borderBottom: '1px solid var(--color-rule)' }}
      >
        <div className="max-w-3xl mx-auto px-6 pt-5 pb-4 flex items-end justify-between gap-6">
          <div>
            <div
              className="font-display text-[34px] leading-none tracking-tight"
              style={{ color: 'var(--color-ink)' }}
            >
              Trustworthy Answer Protocol
            </div>
            <div
              className="font-mono text-[10px] uppercase mt-2 tracking-[0.18em]"
              style={{ color: 'var(--color-ink-soft)' }}
            >
              <span style={{ color: 'var(--color-accent)' }}>◆</span> Diagnostic readout
            </div>
          </div>
          <div
            className="font-mono text-[10.5px] text-right leading-snug uppercase tracking-[0.12em] hidden sm:block"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            Calibrated confidence<br />
            Tampering resistance<br />
            Jailbreak resistance
          </div>
        </div>
        <div className="max-w-3xl mx-auto px-6 pb-3">
          <div className="rule-double" />
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 pt-4 pb-10">
          {messages.length === 0 && (
            <div>
              <div
                className="font-mono text-[10px] uppercase tracking-[0.18em] mb-5"
                style={{ color: 'var(--color-accent)' }}
              >
                — Brief
              </div>
              <h2
                className="font-display text-[42px] leading-[1.05] tracking-tight mb-7 max-w-2xl"
                style={{ color: 'var(--color-ink)' }}
              >
                Can you trust what a medical LLM tells you?
              </h2>
              <p
                className="text-[15px] leading-[1.65] max-w-xl mb-4"
                style={{ color: 'var(--color-ink-2)' }}
              >
                Large language models can sound confident while being wrong, manipulated, or
                jailbroken, a serious risk in clinical settings. TAP reports three trust
                signals alongside each response, with a claim-by-claim breakdown available on demand.
              </p>

              <ul
                className="font-mono text-[11px] uppercase tracking-[0.16em] mb-10 space-y-1.5"
                style={{ color: 'var(--color-ink-muted)' }}
              >
                <li>
                  <span style={{ color: 'var(--color-accent)' }}>01 </span>
                  Certainty <span style={{ color: 'var(--color-ink-soft)' }}>— calibrated confidence the answer is correct</span>
                </li>
                <li>
                  <span style={{ color: 'var(--color-accent)' }}>02 </span>
                  Security <span style={{ color: 'var(--color-ink-soft)' }}>— resistance to training-data tampering</span>
                </li>
                <li>
                  <span style={{ color: 'var(--color-accent)' }}>03 </span>
                  Robustness <span style={{ color: 'var(--color-ink-soft)' }}>— resistance to jailbreak prompts</span>
                </li>
              </ul>

              <div
                className="font-mono text-[10px] uppercase tracking-[0.18em] mb-1"
                style={{ color: 'var(--color-accent)' }}
              >
                — Sample queries
              </div>
              <div
                className="font-mono text-[9.5px] uppercase tracking-[0.16em] mb-3"
                style={{ color: 'var(--color-ink-soft)' }}
              >
                Tuned on medical exam-style questions. Free-text or multiple choice both work.
              </div>
              <div className="flex flex-col" style={{ borderTop: '1px solid var(--color-rule-soft)' }}>
                {SAMPLE_QUERIES.map((q, i) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    disabled={loading}
                    className="group text-left flex items-baseline gap-4 py-3 px-1 transition-colors disabled:opacity-40 hover:bg-[color:var(--color-paper-2)]/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
                    style={{ borderBottom: '1px solid var(--color-rule-soft)' }}
                  >
                    <span
                      className="font-mono text-[10.5px] tabular-nums shrink-0"
                      style={{ color: 'var(--color-accent)' }}
                    >
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <span
                      className="text-[13.5px] leading-snug"
                      style={{ color: 'var(--color-ink-2)' }}
                    >
                      {q}
                    </span>
                    <span
                      className="ml-auto font-mono text-[10px] uppercase tracking-wider opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                      style={{ color: 'var(--color-ink-muted)' }}
                    >
                      ↵ run
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => {
            const turn = messages.slice(0, i + 1).filter((m) => m.role === 'user').length;
            return (
              <div key={i} ref={i === messages.length - 1 ? lastMsgRef : undefined}>
                {msg.role === 'user' && turn > 1 && <ExchangeSeparator turn={turn} />}
                <ChatMessage message={msg} />
              </div>
            );
          })}

          {showSkeleton && <ResponseSkeleton active={showSkeleton} />}

          {error && (
            <div
              role="alert"
              className="text-[13px] px-3 py-2 mb-4 font-mono"
              style={{
                color: 'var(--color-bad)',
                background: 'var(--color-bad-soft)',
                border: '1px solid var(--color-bad)',
              }}
            >
              {error}
            </div>
          )}
        </div>
      </div>

      <ChatInput onSubmit={send} loading={loading} />
    </div>
  );
}

export default App;
