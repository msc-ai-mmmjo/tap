import type { CSSProperties } from 'react';
import type { Claim } from '../types/api';
import { HoverCard } from './HoverCard';

// Renders raw_response as plain text with claim ranges wrapped in tooltip
// spans. Markdown is intentionally not rendered here — threading char
// offsets through a markdown AST is non-trivial, and the right approach
// depends on the shape of claims the real decomposer returns (sub-sentence,
// possibly paraphrased, possibly overlapping). Revisit once that contract
// is known rather than investing against the mock.

interface Props {
  text: string;
  claims: Claim[];
}

type Segment =
  | { kind: 'text'; content: string }
  | { kind: 'claim'; content: string; claim: Claim };

// Splits the raw response into an ordered list of plain-text and claim
// segments. Claims that overlap a previously emitted claim are dropped —
// the mock decomposer never produces these, but the real one may, and the
// resolution strategy is a design decision we defer until we see the shape
// the olmo_tap decomposer returns.
function buildSegments(text: string, claims: Claim[]): Segment[] {
  const ranges = claims
    .filter((c) => c.end > c.start && c.start >= 0 && c.end <= text.length)
    .sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;

  for (const claim of ranges) {
    if (claim.start < cursor) continue;
    if (claim.start > cursor) {
      segments.push({ kind: 'text', content: text.slice(cursor, claim.start) });
    }
    segments.push({
      kind: 'claim',
      content: text.slice(claim.start, claim.end),
      claim,
    });
    cursor = claim.end;
  }
  if (cursor < text.length) {
    segments.push({ kind: 'text', content: text.slice(cursor) });
  }
  return segments;
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

function InlineClaim({ claim, content }: { claim: Claim; content: string }) {
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
          style={{ ...underlineStyle(claim.confidence_level), cursor: 'help' }}
          className="rounded-[1px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
        >
          {content}
        </span>
      )}
    </HoverCard>
  );
}

export function InlineClaims({ text, claims }: Props) {
  const segments = buildSegments(text, claims);
  return (
    <div
      className="text-[15px] leading-[1.7]"
      style={{ color: 'var(--color-ink-2)', whiteSpace: 'pre-wrap' }}
    >
      {segments.map((seg, i) => {
        if (seg.kind === 'text') return <span key={i}>{seg.content}</span>;
        if (seg.claim.confidence_level === 'high') {
          return <span key={i}>{seg.content}</span>;
        }
        return <InlineClaim key={i} claim={seg.claim} content={seg.content} />;
      })}
    </div>
  );
}
