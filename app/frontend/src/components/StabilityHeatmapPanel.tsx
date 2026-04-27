import type { SecurityStatus } from '../types/api';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

const MAX_ALPHA = 0.55;
const N_STEPS = 9; // radii 0–8

// radius 0 (unstable) → brightest; radius 8 (stable) → transparent
const RADIUS_COLORS = Array.from({ length: N_STEPS }, (_, r) =>
  `rgba(var(--color-accent-rgb), ${((1 - r / (N_STEPS - 1)) * MAX_ALPHA).toFixed(3)})`,
);

function bgForRadius(radius: number): string {
  return RADIUS_COLORS[Math.max(0, Math.min(N_STEPS - 1, Math.round(radius)))];
}

const LEGEND_GRADIENT = (() => {
  const w = 100 / N_STEPS;
  return RADIUS_COLORS.map(
    (c, i) => `${c} ${(i * w).toFixed(2)}%,${c} ${((i + 1) * w).toFixed(2)}%`,
  ).join(',');
})();

export function StabilityHeatmapPanel({ data }: Props) {
  if (!data.stability_radii) return null;

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
              background: `linear-gradient(to right, ${LEGEND_GRADIENT})`,
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
          const radius = data.stability_radii![i] ?? 0;

          return (
            <span key={i}>
              <TokenTooltip
                token={tok}
                tooltipBody={<>radius: {radius}</>}
                triggerStyle={{
                  background: bgForRadius(radius),
                  padding: '1px 2px',
                  borderRadius: '2px',
                }}
              />
              {i < data.tokens.length - 1 ? ' ' : ''}
            </span>
          );
        })}
      </div>
    </div>
  );
}
