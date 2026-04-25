import type { SecurityStatus, SecurityResample } from '../types/api';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

const MAX_ALPHA = 0.55;

function bgFor(severity: number | undefined): string {
  if (severity === undefined) return 'transparent';
  const clamped = Math.max(0, Math.min(1, severity));
  return `rgba(var(--color-accent-rgb), ${(clamped * MAX_ALPHA).toFixed(3)})`;
}

function HeatmapToken({
  token,
  resample,
}: {
  token: string;
  resample: SecurityResample;
}) {
  return (
    <TokenTooltip
      token={token}
      tooltipBody={
        <>
          <div>{resample.old_token} → {resample.new_token}</div>
          <div style={{ opacity: 0.7 }}>
            severity {resample.severity.toFixed(2)}
          </div>
        </>
      }
      triggerStyle={{
        background: bgFor(resample.severity),
        padding: '1px 2px',
        borderRadius: '2px',
      }}
    />
  );
}

export function TokenHeatmapPanel({ data }: Props) {
  const resampleByIndex = new Map<number, SecurityResample>(
    data.resampled.map((r) => [r.index, r]),
  );

  return (
    <div className="mt-5">
      <div
        className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
        style={{ color: 'var(--color-ink-muted)' }}
      >
        <span>— Token uncertainty heatmap</span>
        <span style={{ color: 'var(--color-ink-soft)' }}>
          {data.resampled.length} / {data.tokens.length} tokens flagged
        </span>
      </div>
      {data.resampled.length === 0 && (
        <div
          className="text-[11.5px] italic mb-2"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          No uncertain tokens.
        </div>
      )}
      <div
        className="font-mono text-[13px] leading-[1.9]"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        {data.tokens.map((tok, i) => {
          const r = resampleByIndex.get(i);
          return (
            <span key={i}>
              {r ? (
                <HeatmapToken token={tok} resample={r} />
              ) : (
                tok
              )}
              {i < data.tokens.length - 1 ? ' ' : ''}
            </span>
          );
        })}
      </div>
    </div>
  );
}
