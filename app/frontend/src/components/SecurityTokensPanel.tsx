import type { SecurityStatus, SecurityResample } from '../types/api';
import { VALIDITY_RADIUS_MAX } from '../lib/security';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

const MAX_ALPHA = 0.55;
const MIN_ALPHA = 0.05;
const MAX_RADIUS = VALIDITY_RADIUS_MAX;
const N_STEPS = MAX_RADIUS + 1;

// radius 0 (low validity) → brightest; radius 5 (high validity) → faint but visible
const VALIDITY_COLORS = Array.from({ length: N_STEPS }, (_, r) =>
  `rgba(var(--color-accent-rgb), ${(MIN_ALPHA + (1 - r / (N_STEPS - 1)) * (MAX_ALPHA - MIN_ALPHA)).toFixed(3)})`,
);

function bgForValidityRadius(radius: number | undefined): string {
  if (radius == null) return 'transparent';
  return VALIDITY_COLORS[Math.max(0, Math.min(N_STEPS - 1, Math.round(radius)))];
}

// Legend runs high validity (dim) → low validity (bright), so reverse the colour order
const LEGEND_GRADIENT = (() => {
  const reversed = [...VALIDITY_COLORS].reverse();
  const w = 100 / N_STEPS;
  return reversed
    .map((c, i) => `${c} ${(i * w).toFixed(2)}%,${c} ${((i + 1) * w).toFixed(2)}%`)
    .join(',');
})();

function ResampledToken({ token, resample }: { token: string; resample: SecurityResample }) {
  return (
    <TokenTooltip
      token={token}
      tooltipBody={
        <>
          {resample.old_token} → {resample.new_token}
          {resample.validity_radius != null && (
            <> · validity radius: {resample.validity_radius} / {N_STEPS - 1}</>
          )}
        </>
      }
      triggerStyle={{
        background: bgForValidityRadius(resample.validity_radius),
        padding: '1px 2px',
        borderRadius: '2px',
        textDecoration: 'underline',
        textDecorationStyle: 'solid',
        textUnderlineOffset: '3px',
      }}
    />
  );
}

export function SecurityTokensPanel({ data }: Props) {
  const resampleByIndex = new Map<number, SecurityResample>(
    data.resampled.map((r) => [r.index, r]),
  );

  return (
    <div className="mt-5 animate-fade-in">
      <div
        className="px-5 pt-4 pb-4"
        style={{
          background: 'var(--color-card)',
          border: '1px solid var(--color-rule)',
        }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] flex items-center justify-between gap-3"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Security token stream</span>
          <span
            className="flex items-center gap-2"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            <span>high validity</span>
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
            <span>low validity</span>
          </span>
        </div>
        <div
          className="text-[11px] mb-3 mt-1"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          Underlined tokens were resampled; the shade encodes their validity
          radius (0–{MAX_RADIUS}, higher is safer).
        </div>
        {data.resampled.length === 0 && (
          <div
            className="text-[11.5px] italic mb-2"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            No resampled tokens.
          </div>
        )}
        <div
          className="font-mono text-[13px] leading-[1.7]"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          {data.tokens.map((tok, i) => {
            const r = resampleByIndex.get(i);
            return (
              <span key={i}>
                {r ? <ResampledToken token={tok} resample={r} /> : tok}
                {i < data.tokens.length - 1 ? ' ' : ''}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
