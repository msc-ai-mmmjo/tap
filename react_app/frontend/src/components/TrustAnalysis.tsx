import type { AnalysisResponse } from '../types/api';
import { ClaimCard } from './ClaimCard';

interface Props {
  data: AnalysisResponse;
}

export function TrustAnalysis({ data }: Props) {
  return (
    <div className="mt-3 animate-fade-in">
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <span className="text-[13px] font-medium text-gray-500 uppercase tracking-wide">
            Claim analysis
          </span>
        </div>
        {data.claims.map((claim, i) => (
          <ClaimCard key={i} claim={claim} />
        ))}
      </div>
    </div>
  );
}
