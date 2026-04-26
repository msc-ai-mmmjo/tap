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
      entropy {entropy.toFixed(2)} nats
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
    for (const e of data.token_entropies ?? []) if (e > m) m = e;
    return m;
  }, [data.token_entropies]);

  const flagged = useMemo(() => {
    if (maxEntropy <= 0) return 0;
    const entropies = data.token_entropies ?? [];
    // Bound by tokens.length so the counter can't claim more highlights than
    // there are tokens to paint, in case the parallel arrays ever drift.
    const limit = Math.min(data.tokens.length, entropies.length);
    let n = 0;
    for (let i = 0; i < limit; i++) {
      if (entropies[i] / maxEntropy >= VISIBILITY_FLOOR) n += 1;
    }
    return n;
  }, [data.token_entropies, data.tokens.length, maxEntropy]);

  return (
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
        <span>— Token uncertainty heatmap</span>
        <span style={{ color: 'var(--color-ink-soft)' }}>
          {flagged} / {data.tokens.length} tokens highlighted
        </span>
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
