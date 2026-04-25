import { useMemo } from 'react';
import type { SecurityStatus, SecurityResample } from '../types/api';
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
  resample,
}: {
  token: string;
  intensity: number;
  entropy: number | undefined;
  resample: SecurityResample | undefined;
}) {
  const triggerStyle = {
    background: bgForIntensity(intensity),
    padding: '1px 2px',
    borderRadius: '2px',
  };
  const tooltipBody = (
    <>
      {resample && (
        <div>{resample.old_token} → {resample.new_token}</div>
      )}
      {entropy !== undefined && (
        <div style={{ opacity: 0.7 }}>
          entropy {entropy.toFixed(2)} nats
        </div>
      )}
    </>
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
  const resampleByIndex = useMemo(
    () =>
      new Map<number, SecurityResample>(
        data.resampled.map((r) => [r.index, r]),
      ),
    [data.resampled],
  );

  const maxEntropy = useMemo(() => {
    let m = 0;
    for (const e of data.token_entropies) if (e > m) m = e;
    return m;
  }, [data.token_entropies]);

  const flagged = useMemo(() => {
    if (maxEntropy <= 0) return 0;
    let n = 0;
    for (const e of data.token_entropies) {
      if (e / maxEntropy >= VISIBILITY_FLOOR) n += 1;
    }
    return n;
  }, [data.token_entropies, maxEntropy]);

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
          const entropy = data.token_entropies[i];
          const intensity =
            maxEntropy > 0 && entropy !== undefined ? entropy / maxEntropy : 0;
          const r = resampleByIndex.get(i);
          const visible = intensity >= VISIBILITY_FLOOR;
          return (
            <span key={i}>
              {visible || r ? (
                <HeatmapToken
                  token={tok}
                  intensity={visible ? intensity : 0}
                  entropy={entropy}
                  resample={r}
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
