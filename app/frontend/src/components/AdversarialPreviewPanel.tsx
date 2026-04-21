import type { RobustnessStatus } from '../types/api';

interface Props {
  data: RobustnessStatus;
}

export function AdversarialPreviewPanel({ data }: Props) {
  const body = data.attacked_response;

  return (
    <div className="mt-5 animate-fade-in">
      <div
        className="px-5 pt-4 pb-4"
        style={{
          background: 'var(--color-paper-2)',
          border: '1px solid var(--color-rule)',
        }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Adversarial response</span>
          <span
            className="px-2 py-0.5 font-mono text-[9.5px] tracking-[0.14em]"
            style={{
              color: 'var(--color-paper)',
              background: 'var(--color-warn)',
            }}
          >
            WIP · PREVIEW
          </span>
        </div>

        {data.type === 'mcq' && (
          <div
            className="font-mono text-[12.5px] mb-3"
            style={{ color: 'var(--color-ink)' }}
          >
            <span style={{ color: 'var(--color-ink-muted)' }}>Original </span>
            <span>{data.original_choice}</span>
            <span style={{ color: 'var(--color-ink-muted)' }}> → Attacked </span>
            <span
              style={{
                color: data.flipped ? 'var(--color-bad)' : 'var(--color-ok)',
              }}
            >
              {data.attacked_choice}
            </span>
          </div>
        )}

        {body ? (
          <div
            className="text-[13.5px] leading-[1.65] whitespace-pre-wrap"
            style={{ color: 'var(--color-ink-2)' }}
          >
            {body}
          </div>
        ) : (
          <div
            className="text-[11.5px] italic"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            (no adversarial response generated)
          </div>
        )}
      </div>
    </div>
  );
}
