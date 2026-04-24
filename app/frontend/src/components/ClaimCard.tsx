import Markdown from 'react-markdown';
import type { Claim } from '../types/api';
import { getConfidenceStyle } from '../lib/confidence';

interface Props {
  claim: Claim;
  index: number;
}

export function ClaimCard({ claim, index }: Props) {
  const style = getConfidenceStyle(claim.confidence);
  const pct = Math.round(claim.confidence * 100);

  return (
    <li
      className="grid gap-4 py-3.5"
      style={{
        gridTemplateColumns: 'auto 1fr auto',
        borderTop: '1px solid var(--color-rule-soft)',
      }}
    >
      <div
        className="font-mono text-[10.5px] tabular-nums pt-1"
        style={{ color: 'var(--color-accent-warm)' }}
      >
        {String(index).padStart(2, '0')}
      </div>

      <div className="min-w-0">
        <div
          className="text-[14px] leading-[1.6] prose prose-sm max-w-none prose-p:my-0 prose-strong:font-semibold"
          style={{ color: 'var(--color-ink)' }}
        >
          <Markdown>{claim.text}</Markdown>
        </div>
        {claim.guidance && (
          <div
            className="text-[11.5px] mt-1.5"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            ⚠ {claim.guidance}
          </div>
        )}
      </div>

      <div className="flex flex-col items-end gap-1.5 pt-1 shrink-0">
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-1 h-3.5"
            style={{ background: style.bar }}
            aria-hidden
          />
          <span
            className="font-mono text-[12px] tabular-nums font-medium"
            style={{ color: style.pillText }}
          >
            {pct}%
          </span>
        </div>
        <div
          className="font-mono text-[9.5px] uppercase tracking-[0.14em]"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          Confidence
        </div>
      </div>
    </li>
  );
}
