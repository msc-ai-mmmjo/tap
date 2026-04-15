import type { Element, ElementContent, Root, Text } from 'hast';
import type { Plugin } from 'unified';
import type { Claim } from '../types/api';

// rehype plugin: wrap text node ranges that match claim offsets in a
// <mark> carrying the claim index. Runs after markdown → hast, so we
// never touch the source string and can't corrupt syntax. Claim offsets
// are compared against text node `position.offset` values set by
// remark-parse, so markdown structure (bold, lists, links) is preserved
// intact around and inside the wrapped range.
//
// Claims that straddle structural boundaries (e.g. a sentence with
// **bold** inside it) will produce multiple adjacent <mark> segments —
// one per text node the claim overlaps. Each carries the same claim
// index and so renders the same tooltip; visually the underlines sit
// flush, which is acceptable. Overlapping claims against each other are
// resolved first-wins per text node.

export function rehypeClaimMarks(claims: Claim[]): Plugin<[], Root> {
  const visible = claims
    .map((claim, idx) => ({ claim, idx }))
    .filter(
      ({ claim }) =>
        claim.confidence_level !== 'high' && claim.end > claim.start,
    );

  return () => (tree) => {
    if (visible.length === 0) return;
    transform(tree, visible);
  };
}

interface IndexedClaim {
  claim: Claim;
  idx: number;
}

function transform(
  parent: Root | Element,
  claims: IndexedClaim[],
): void {
  const next: ElementContent[] = [];
  for (const child of parent.children) {
    if (child.type === 'text') {
      next.push(...splitTextNode(child, claims));
    } else if (child.type === 'element') {
      transform(child, claims);
      next.push(child);
    } else {
      next.push(child as ElementContent);
    }
  }
  parent.children = next as typeof parent.children;
}

function splitTextNode(node: Text, claims: IndexedClaim[]): ElementContent[] {
  const nodeStart = node.position?.start?.offset;
  const nodeEnd = node.position?.end?.offset;
  if (nodeStart === undefined || nodeEnd === undefined) return [node];

  const overlapping = claims
    .filter(({ claim }) => claim.end > nodeStart && claim.start < nodeEnd)
    .sort((a, b) => a.claim.start - b.claim.start);

  if (overlapping.length === 0) return [node];

  const out: ElementContent[] = [];
  let cursor = nodeStart;

  for (const { claim, idx } of overlapping) {
    const start = Math.max(claim.start, nodeStart);
    const end = Math.min(claim.end, nodeEnd);
    if (start < cursor) continue;
    if (start > cursor) {
      out.push(makeText(node.value.slice(cursor - nodeStart, start - nodeStart)));
    }
    out.push({
      type: 'element',
      tagName: 'mark',
      properties: { claimIdx: String(idx) },
      children: [makeText(node.value.slice(start - nodeStart, end - nodeStart))],
    });
    cursor = end;
  }
  if (cursor < nodeEnd) {
    out.push(makeText(node.value.slice(cursor - nodeStart)));
  }
  return out;
}

function makeText(value: string): Text {
  return { type: 'text', value };
}
