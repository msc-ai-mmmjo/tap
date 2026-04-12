import Markdown from 'react-markdown';
import type { Claim } from '../types/api';
import { getConfidenceStyle } from '../lib/confidence';

interface Props {
  claim: Claim;
}

export function ClaimCard({ claim }: Props) {
  const style = getConfidenceStyle(claim.confidence);

  return (
    <div className="flex mb-3.5 rounded-lg px-2 py-2 -mx-2 transition-colors hover:bg-gray-50/80">
      <div
        className="shrink-0 rounded-sm"
        style={{ width: 4, backgroundColor: style.bar }}
      />
      <div className="ml-3 min-w-0">
        <div className="text-[14px] leading-[1.6] text-gray-800 mb-1 prose prose-sm max-w-none prose-p:my-0 prose-strong:text-gray-800">
          <Markdown>{claim.text}</Markdown>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="text-[11px] px-2 py-0.5 rounded-md font-medium"
            style={{ backgroundColor: style.pillBg, color: style.pillText }}
          >
            P(correct): {claim.confidence.toFixed(2)}
          </span>
          {claim.guidance && (
            <span className="text-[11px] text-gray-400">{claim.guidance}</span>
          )}
        </div>
      </div>
    </div>
  );
}
