import type { RobustnessStatus } from '../types/api';

interface Props {
  data: RobustnessStatus;
}

export function AdversarialPreviewPanel({ data }: Props) {
  const isUnavailable = data.type === 'unavailable';
  const pillLabel = isUnavailable ? '[WIP · PENDING]' : '[WIP · PREVIEW]';
  const pillColour = isUnavailable ? 'var(--color-ink-muted)' : 'var(--color-warn)';

  return (
    <div className="mt-4">
      <div
        className="px-5 pt-4 pb-4"
        style={{
          background: 'var(--color-card)',
          border: '1px solid var(--color-rule)',
        }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Adversarial preview</span>
          <span
            className="px-2 py-0.5 text-[9.5px]"
            style={{ color: pillColour, border: `1px solid ${pillColour}` }}
          >
            {pillLabel}
          </span>
        </div>

        {isUnavailable ? (
          <div className="text-[12px]" style={{ color: 'var(--color-ink-soft)' }}>
            Robustness pipeline is being wired, check back soon.
          </div>
        ) : (
          <ActiveBody data={data} />
        )}
      </div>
    </div>
  );
}

function ActiveBody({
  data,
}: {
  data: Exclude<RobustnessStatus, { type: 'unavailable' }>;
}) {
  const hasResponse = data.attacked_response.length > 0;
  const hasSuffix = data.attack_suffix.length > 0;

  return (
    <>
      <div className="text-[12px] mb-3" style={{ color: 'var(--color-ink-soft)' }}>
        {data.flipped} of {data.attempts} attacks flipped the answer
      </div>

      {data.type === 'mcq' && (
        <div className="mb-3 text-[12px] font-mono">
          <div style={{ color: 'var(--color-ink-muted)' }}>
            Original:{' '}
            <span style={{ color: 'var(--color-ink)' }}>
              {data.original_choice || '—'}
            </span>
          </div>
          <div style={{ color: 'var(--color-ink-muted)' }}>
            Under attack:{' '}
            <span
              style={{
                color:
                  data.attacked_choice &&
                  data.attacked_choice !== data.original_choice
                    ? 'var(--color-bad)'
                    : 'var(--color-ink-muted)',
              }}
            >
              {data.attacked_choice || '—'}
            </span>
          </div>
        </div>
      )}

      {hasSuffix && (
        <div className="mb-3">
          <div
            className="font-mono text-[10px] uppercase tracking-[0.16em] mb-1"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            — Attack suffix
          </div>
          <div
            className="font-mono text-[11.5px]"
            style={{
              color: 'var(--color-ink-soft)',
              wordBreak: 'break-all',
              overflowWrap: 'anywhere',
            }}
          >
            {data.attack_suffix}
          </div>
        </div>
      )}

      <div>
        <div
          className="font-mono text-[10px] uppercase tracking-[0.16em] mb-1"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          — Adversarial response
        </div>
        {hasResponse ? (
          <div
            className="text-[12.5px]"
            style={{
              color: 'var(--color-ink)',
              background: 'var(--color-paper-2)',
              padding: '8px 10px',
              whiteSpace: 'pre-wrap',
            }}
          >
            {data.attacked_response}
          </div>
        ) : (
          <div className="text-[12px]" style={{ color: 'var(--color-ink-soft)' }}>
            (no adversarial response captured)
          </div>
        )}
      </div>
    </>
  );
}
