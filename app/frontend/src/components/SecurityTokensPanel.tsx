import type { SecurityStatus, SecurityResample } from '../types/api';
import { TokenTooltip } from './TokenTooltip';

interface Props {
  data: SecurityStatus;
}

function ResampledToken({ token, resample }: { token: string; resample: SecurityResample }) {
  return (
    <TokenTooltip
      token={token}
      tooltipBody={<>{resample.old_token} → {resample.new_token}</>}
      triggerStyle={{
        color: 'var(--color-accent)',
        borderBottom: '1px dotted var(--color-accent)',
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
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Security token stream</span>
          <span style={{ color: 'var(--color-ink-soft)' }}>
            {data.resampled.length} / {data.tokens.length} resampled
          </span>
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
