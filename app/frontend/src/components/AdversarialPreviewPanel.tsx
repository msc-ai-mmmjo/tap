import type { AdversarialWorstCase, RobustnessStatus } from '../types/api';

interface Props {
  data: RobustnessStatus;
}

export function AdversarialPreviewPanel({ data }: Props) {
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
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          — Adversarial preview
        </div>

        {data.type === 'unavailable' ? (
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
  return (
    <>
      <div className="text-[12px] mb-3" style={{ color: 'var(--color-ink-soft)' }}>
        {data.flipped} of {data.attempts} attacks flipped the answer
        {data.worst_case && !data.worst_case.flipped
          ? '. Showing the closest attempt.'
          : ''}
      </div>

      {data.worst_case ? (
        <WorstCaseBody worstCase={data.worst_case} isMcq={data.type === 'mcq'} />
      ) : (
        <div className="text-[12px]" style={{ color: 'var(--color-ink-soft)' }}>
          No adversarial example to preview. The answer held across every attack.
        </div>
      )}
    </>
  );
}

function WorstCaseBody({
  worstCase,
  isMcq,
}: {
  worstCase: AdversarialWorstCase;
  isMcq: boolean;
}) {
  const flippedColour = worstCase.flipped ? 'var(--color-bad)' : 'var(--color-ink-muted)';

  return (
    <>
      {isMcq && (
        <div className="mb-3 text-[12px] font-mono">
          <div style={{ color: 'var(--color-ink-muted)' }}>
            Original:{' '}
            <span style={{ color: 'var(--color-ink)' }}>
              {worstCase.clean_response || '—'}
            </span>
          </div>
          <div style={{ color: 'var(--color-ink-muted)' }}>
            Under attack:{' '}
            <span style={{ color: flippedColour }}>
              {worstCase.adv_response || '—'}
            </span>
          </div>
        </div>
      )}

      {worstCase.suffix && (
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
            {worstCase.suffix}
          </div>
        </div>
      )}

      {!isMcq && (
        <div>
          <div
            className="font-mono text-[10px] uppercase tracking-[0.16em] mb-1"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            {worstCase.flipped ? '— Adversarial response' : '— Response under this attack'}
          </div>
          {worstCase.adv_response ? (
            <div
              className="text-[12.5px]"
              style={{
                color: 'var(--color-ink)',
                background: 'var(--color-paper-2)',
                padding: '8px 10px',
                whiteSpace: 'pre-wrap',
              }}
            >
              {worstCase.adv_response}
            </div>
          ) : (
            <div className="text-[12px]" style={{ color: 'var(--color-ink-soft)' }}>
              (no adversarial response captured)
            </div>
          )}
        </div>
      )}
    </>
  );
}
