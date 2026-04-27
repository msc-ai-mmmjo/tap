import { useMemo } from 'react';
import type { SecurityStatus, SecurityResample } from '../types/api';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

const MAX_ALPHA = 0.55;

function bgForMargin(margin: number): string {
  const clamped = Math.max(0, Math.min(1, margin));
  // Low margin (contested) → bright; high margin (stable) → dim
  return `rgba(var(--color-accent-rgb), ${((1 - clamped) * MAX_ALPHA).toFixed(3)})`;
}

export function StabilityHeatmapPanel({ data }: Props) {
  if (!data.stability_radii || !data.stability_margins) return null;

  const rejectionByIndex = useMemo(() => {
    const m = new Map<number, SecurityResample>();
    for (const r of data.resampled) m.set(r.index, r);
    return m;
  }, [data.resampled]);

  return (
    <div
      className="px-5 pt-4 pb-4"
      style={{
        background: 'var(--color-card)',
        border: '1px solid var(--color-rule)',
      }}
    >
      <div
        className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between gap-3"
        style={{ color: 'var(--color-ink-muted)' }}
      >
        <span>— Token stability heatmap</span>
        <span
          className="flex items-center gap-2"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          <span>more stable</span>
          <span
            aria-hidden
            style={{
              display: 'inline-block',
              width: '64px',
              height: '8px',
              background: `linear-gradient(to right, rgba(var(--color-accent-rgb), 0), rgba(var(--color-accent-rgb), ${MAX_ALPHA}))`,
              border: '1px solid var(--color-rule)',
            }}
          />
          <span>less stable</span>
        </span>
      </div>
      <div
        className="font-mono text-[13px] leading-[1.9]"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        {data.tokens.map((tok, i) => {
          const margin = data.stability_margins![i] ?? 0;
          const radius = data.stability_radii![i] ?? 0;
          const rejection = rejectionByIndex.get(i);

          const chipStyle: React.CSSProperties = {
            background: bgForMargin(margin),
            padding: '1px 2px',
            borderRadius: '2px',
            ...(rejection
              ? {
                  textDecoration: 'underline',
                  textDecorationStyle: 'dotted',
                  textUnderlineOffset: '3px',
                }
              : {}),
          };

          const tooltipBody = rejection ? (
            <>
              radius: {radius} · suppression:{' '}
              {rejection.suppression_score?.toFixed(3) ?? '—'} · {rejection.old_token} → {tok}
            </>
          ) : (
            <>radius: {radius}</>
          );

          return (
            <span key={i}>
              <TokenTooltip
                token={tok}
                tooltipBody={tooltipBody}
                triggerStyle={chipStyle}
              />
              {i < data.tokens.length - 1 ? ' ' : ''}
            </span>
          );
        })}
      </div>
    </div>
  );
}
