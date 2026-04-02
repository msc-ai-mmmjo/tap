import type { AnalysisResponse } from '../types/api';

interface Props {
  data: AnalysisResponse;
}

export function MetricCards({ data }: Props) {
  const securityValue = data.security.certified ? 'Certified' : 'Caution';
  const robustnessValue = data.robustness.passed ? 'Passed' : 'Failed';
  const greenColor = '#1D9E75';

  return (
    <div className="grid grid-cols-3 gap-2.5">
      <div className="bg-gray-50 rounded-lg p-3">
        <div className="text-[11px] uppercase tracking-wide text-gray-400 mb-1">
          Uncertainty
        </div>
        <div className="text-[18px] font-medium text-gray-800">
          {data.overall_confidence.toFixed(2)}
        </div>
        <div className="text-[11px] text-gray-400 mt-0.5">LoRA probe avg</div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <div className="text-[11px] uppercase tracking-wide text-gray-400 mb-1">
          Security
        </div>
        <div
          className="text-[18px] font-medium"
          style={{ color: data.security.certified ? greenColor : '#EF9F27' }}
        >
          {securityValue}
        </div>
        <div className="text-[11px] text-gray-400 mt-0.5">
          TPA budget: {data.security.tpa_budget ?? '—'} samples
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <div className="text-[11px] uppercase tracking-wide text-gray-400 mb-1">
          Robustness
        </div>
        <div
          className="text-[18px] font-medium"
          style={{ color: data.robustness.passed ? greenColor : '#E24B4A' }}
        >
          {robustnessValue}
        </div>
        <div className="text-[11px] text-gray-400 mt-0.5">
          {data.robustness.detail}
        </div>
      </div>
    </div>
  );
}
