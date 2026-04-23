import type { AnalysisResponse } from '../types/api';
import { ClaimCard } from './ClaimCard';
import { SecurityTokensPanel } from './SecurityTokensPanel';

interface Props {
  data: AnalysisResponse;
}

export function TrustAnalysis({ data }: Props) {
  return (
    <div className="mt-5 animate-fade-in">
      <div
        className="px-5 pt-4 pb-2"
        style={{
          background: 'var(--color-card)',
          border: '1px solid var(--color-rule)',
        }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Claim ledger</span>
          <span style={{ color: 'var(--color-ink-soft)' }}>
            {data.claims.length} {data.claims.length === 1 ? 'claim' : 'claims'}
          </span>
        </div>
        <ol>
          {data.claims.map((claim, i) => (
            <ClaimCard key={i} claim={claim} index={i + 1} />
          ))}
        </ol>
      </div>

      {data.security.tokens.length > 0 && (
        <SecurityTokensPanel data={data.security} />
      )}
    </div>
  );
}
