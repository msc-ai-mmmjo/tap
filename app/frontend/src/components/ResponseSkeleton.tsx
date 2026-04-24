import { useCyclingPhrase } from '../hooks/useCyclingPhrase';

const PHASES = [
  'Generating response',
  'Certifying tokens',
  'Scoring uncertainty',
  'Probing robustness',
  'Decomposing claims',
] as const;

const GHOST_BAR_WIDTHS = ['92%', '87%', '95%', '45%'];
const GHOST_CARD_LABELS = ['Certainty', 'Security', 'Robustness'];

interface Props {
  active: boolean;
}

export function ResponseSkeleton({ active }: Props) {
  const phrase = useCyclingPhrase(PHASES, 1800, active);

  return (
    <div
      className="mb-10 animate-fade-in"
      role="status"
      aria-label="Loading response"
    >
      <div
        className="flex items-center justify-between mb-3"
        style={{ borderTop: '1px solid var(--color-rule)', paddingTop: 8 }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em]"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          — Response
        </div>
        <div
          className="font-mono text-[10px] uppercase tracking-[0.16em] flex items-center gap-2"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          <span
            className="inline-block w-2 h-2 rounded-full animate-pulse motion-reduce:animate-none"
            style={{ background: 'var(--color-accent)' }}
            aria-hidden
          />
          <span
            key={phrase}
            className="phrase-swap"
            aria-live="polite"
            aria-atomic="true"
          >
            {phrase}
          </span>
        </div>
      </div>

      <div className="space-y-3" aria-hidden>
        {GHOST_BAR_WIDTHS.map((w, i) => (
          <div
            key={i}
            className="skeleton-bar"
            style={{ width: w, height: '17px' }}
          />
        ))}
      </div>

      <div
        className="mt-6 grid grid-cols-3"
        style={{
          background: 'var(--color-card)',
          border: '1px solid var(--color-rule-soft)',
        }}
        aria-hidden
      >
        {GHOST_CARD_LABELS.map((label, i) => (
          <div
            key={label}
            className="px-5 py-4"
            style={{
              borderLeft: i === 0 ? 'none' : '1px solid var(--color-rule)',
            }}
          >
            <div
              className="flex items-center justify-between mb-3"
            >
              <div
                className="font-mono text-[10px] uppercase tracking-[0.16em] flex items-baseline gap-1.5"
                style={{ color: 'var(--color-ink-muted)' }}
              >
                <span style={{ color: 'var(--color-accent)' }}>
                  {String(i + 1).padStart(2, '0')}
                </span>
                {label}
              </div>
            </div>
            <div
              className="font-mono text-[18px] tabular-nums leading-none"
              style={{ color: 'var(--color-ink-muted)' }}
            >
              —
            </div>
            <div
              className="skeleton-bar mt-2"
              style={{ width: '70%', height: '16px' }}
            />
          </div>
        ))}
      </div>

      <div
        className="mt-6 font-mono text-[10.5px] uppercase tracking-[0.16em] inline-flex items-center gap-2 py-1.5 opacity-40"
        style={{ color: 'var(--color-ink-muted)' }}
        aria-hidden
      >
        <span
          aria-hidden
          className="inline-block w-3"
          style={{ borderTop: '1px solid var(--color-ink-muted)' }}
        />
        Inspect details
        <span
          aria-hidden
          className="inline-block w-3 text-center leading-none"
        >
          +
        </span>
      </div>
    </div>
  );
}
