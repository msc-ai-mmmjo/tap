import type { AnalysisResponse } from '../types/api';
import { ClaimCard } from './ClaimCard';
import { AdversarialPreviewPanel } from './AdversarialPreviewPanel';
import { SecurityTokensPanel } from './SecurityTokensPanel';
import { TokenHeatmapPanel } from './TokenHeatmapPanel';

interface Props {
  data: AnalysisResponse;
}

export function TrustAnalysis({ data }: Props) {
  const showExperimental =
    data.claims.length > 0 || data.security.tokens.length > 0;

  return (
    <div className="mt-5 animate-fade-in">
      {data.security.tokens.length > 0 && (
        <SecurityTokensPanel data={data.security} />
      )}

      <AdversarialPreviewPanel data={data.robustness} />

      {showExperimental && (
        <div className="mt-8">
          <div
            className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center gap-2"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            <span style={{ color: 'var(--color-accent)' }}>◆</span>
            <span>— Experimental</span>
            <span style={{ color: 'var(--color-ink-soft)' }}>
              · uncertainty
            </span>
          </div>
          <div className="flex flex-col gap-4">
            {data.claims.length > 0 && (
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
            )}

            {data.security.tokens.length > 0 && (
              <TokenHeatmapPanel data={data.security} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
