import type { CSSProperties, ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import type { Claim } from '../types/api';
import { rehypeClaimMarks } from '../lib/claimMarks';
import { HoverCard } from './HoverCard';

// Renders raw_response as markdown with low/moderate confidence claim
// ranges wrapped in a tooltip-bearing <mark>. The wrapping is done by
// a rehype plugin (see lib/claimMarks.ts) which walks the hast AST
// after parsing and splits text nodes at claim offsets — so markdown
// structure (bold, lists, links, headings) renders untouched and claim
// offsets can never corrupt syntax.

interface Props {
  text: string;
  claims: Claim[];
}

function underlineStyle(level: Claim['confidence_level']): CSSProperties {
  if (level === 'moderate') {
    return {
      textDecorationLine: 'underline',
      textDecorationStyle: 'dashed',
      textDecorationColor: 'var(--color-warn)',
      textDecorationThickness: '1.5px',
      textUnderlineOffset: '4px',
    };
  }
  if (level === 'low') {
    return {
      textDecorationLine: 'underline',
      textDecorationStyle: 'solid',
      textDecorationColor: 'var(--color-bad)',
      textDecorationThickness: '2px',
      textUnderlineOffset: '4px',
    };
  }
  return {};
}

function ClaimMark({ claim, children }: { claim: Claim; children: ReactNode }) {
  const pct = Math.round(claim.confidence * 100);
  const accent =
    claim.confidence_level === 'low' ? 'var(--color-bad)' : 'var(--color-warn)';

  return (
    <HoverCard
      width={260}
      content={
        <>
          <div
            className="font-mono text-[10px] uppercase tracking-[0.16em] mb-1.5 flex items-baseline justify-between"
            style={{ color: 'var(--color-paper-2)' }}
          >
            <span>P(correct)</span>
            <span className="tabular-nums font-medium" style={{ color: accent }}>
              {pct}%
            </span>
          </div>
          <div>{claim.guidance}</div>
        </>
      }
    >
      {(trigger) => (
        <span
          {...trigger}
          tabIndex={0}
          role="button"
          aria-label={`Claim, ${pct}% probability of being correct`}
          style={{
            ...underlineStyle(claim.confidence_level),
            cursor: 'help',
            backgroundColor: 'transparent',
          }}
          className="rounded-[1px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
        >
          {children}
        </span>
      )}
    </HoverCard>
  );
}

export function InlineClaims({ text, claims }: Props) {
  const components: Components = {
    mark({ node, children }) {
      // Read the claim index straight off the hast node properties set
      // by rehypeClaimMarks — bypasses any React prop normalization.
      const idxStr = node?.properties?.claimIdx;
      const idx = typeof idxStr === 'string' ? Number(idxStr) : NaN;
      const claim = Number.isInteger(idx) ? claims[idx] : undefined;
      if (!claim) return <>{children}</>;
      return <ClaimMark claim={claim}>{children}</ClaimMark>;
    },
  };

  return (
    <div
      className="text-[15px] leading-[1.7] prose max-w-none prose-p:my-2.5 prose-li:my-1 prose-headings:font-display prose-headings:font-normal prose-strong:font-semibold"
      style={{ color: 'var(--color-ink-2)' }}
    >
      <Markdown
        rehypePlugins={[rehypeClaimMarks(claims)]}
        components={components}
      >
        {text}
      </Markdown>
    </div>
  );
}
