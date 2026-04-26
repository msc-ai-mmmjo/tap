import { useMemo } from 'react';
import type { SecurityStatus } from '../types/api';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

const MAX_ALPHA = 0.55;
// Entropies below this fraction of the response max stay invisible so a
// near-uniform-low response doesn't show as a wall of colour.
const VISIBILITY_FLOOR = 0.05;

function bgForIntensity(intensity: number): string {
  if (intensity <= 0) return 'transparent';
  const clamped = Math.max(0, Math.min(1, intensity));
  return `rgba(var(--color-accent-rgb), ${(clamped * MAX_ALPHA).toFixed(3)})`;
}

function HeatmapToken({
  token,
  intensity,
  entropy,
}: {
  token: string;
  intensity: number;
  entropy: number;
}) {
  const triggerStyle = {
    background: bgForIntensity(intensity),
    padding: '1px 2px',
    borderRadius: '2px',
  };
  const tooltipBody = (
    <div style={{ opacity: 0.7 }}>
      entropy {entropy < 0.005 ? '< 0.01' : entropy.toFixed(2)} nats
    </div>
  );
  return (
    <TokenTooltip
      token={token}
      tooltipBody={tooltipBody}
      triggerStyle={triggerStyle}
    />
  );
}

export function TokenHeatmapPanel({ data }: Props) {
  const maxEntropy = useMemo(() => {
    let m = 0;
    for (const e of data.token_entropies ?? []) if (Number.isFinite(e) && e > m) m = e;
    return m;
  }, [data.token_entropies]);

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
        <span>— Token uncertainty heatmap</span>
        {maxEntropy > 0 && (
          <span
            className="flex items-center gap-2"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            <span>low</span>
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
            <span>high</span>
            <span style={{ opacity: 0.7 }}>
              (max {maxEntropy < 0.005 ? '< 0.01' : maxEntropy.toFixed(2)} nats)
            </span>
          </span>
        )}
      </div>
      {maxEntropy <= 0 && (
        <div
          className="text-[11.5px] italic mb-2"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          No uncertainty signal available.
        </div>
      )}
      <div
        className="font-mono text-[13px] leading-[1.9]"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        {data.tokens.map((tok, i) => {
          const entropy = data.token_entropies?.[i];
          const intensity =
            maxEntropy > 0 && entropy !== undefined ? entropy / maxEntropy : 0;
          const visible = intensity >= VISIBILITY_FLOOR;
          return (
            <span key={i}>
              {visible && entropy !== undefined ? (
                <HeatmapToken
                  token={tok}
                  intensity={intensity}
                  entropy={entropy}
                />
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
