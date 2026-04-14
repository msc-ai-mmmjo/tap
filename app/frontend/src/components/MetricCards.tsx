import type { AnalysisResponse } from '../types/api';
import { COLOURS } from '../lib/constants';

interface Props {
  data: AnalysisResponse;
}

interface MetricInfo {
  definition: string;
  paper: string;
}

const METRIC_INFO: Record<'uncertainty' | 'security' | 'robustness', MetricInfo> = {
  uncertainty: {
    definition:
      "How confident the model is that each claim in its answer is factually correct. Scores near 1.0 mean the model's internal signals match patterns seen in verified-correct answers; low scores mean the claim should be double-checked before acting on it.",
    paper:
      'Method: Kapoor et al., "Language Models Must Be Taught to Know What They Don\'t Know" (NeurIPS 2024).',
  },
  security: {
    definition:
      "Whether the answer can be trusted not to have been shaped by tampered training data. 'Certified' means the response would provably stay the same even if a bounded number of poisoned examples had been slipped into training.",
    paper:
      'Method: "Towards Poisoning Robustness Certification for Natural Language Generation".',
  },
  robustness: {
    definition:
      "Whether the model held its ground against adversarial prompts designed to jailbreak it into unsafe or incorrect output. 'Passed' means known attack strings failed to change the answer.",
    paper:
      'Method: "AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates".',
  },
};

function InfoTooltip({ info }: { info: MetricInfo }) {
  return (
    <span className="relative group inline-flex">
      <svg
        className="w-3.5 h-3.5 text-gray-300 group-hover:text-gray-500 transition-colors cursor-help"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <circle cx="12" cy="12" r="10" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-4M12 8h.01" />
      </svg>
      <span
        role="tooltip"
        className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-gray-900 text-white text-[11px] leading-relaxed rounded-lg shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-150 z-20"
      >
        <span className="block mb-1.5 text-gray-100 normal-case tracking-normal font-normal">
          {info.definition}
        </span>
        <span className="block text-gray-400 italic normal-case tracking-normal font-normal">
          {info.paper}
        </span>
      </span>
    </span>
  );
}

export function MetricCards({ data }: Props) {
  const securityValue = data.security.certified ? 'Certified' : 'Caution';
  const robustnessValue = data.robustness.passed ? 'Passed' : 'Failed';

  return (
    <div className="grid grid-cols-3 gap-2.5">
      <div className="bg-gray-50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-1">
          <div className="text-[11px] uppercase tracking-wide text-gray-400">
            Uncertainty
          </div>
          <InfoTooltip info={METRIC_INFO.uncertainty} />
        </div>
        <div className="text-[18px] font-medium text-gray-800">
          {data.overall_confidence.toFixed(2)}
        </div>
        <div className="text-[11px] text-gray-400 mt-0.5">LoRA probe avg</div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-1">
          <div className="text-[11px] uppercase tracking-wide text-gray-400">
            Security
          </div>
          <InfoTooltip info={METRIC_INFO.security} />
        </div>
        <div
          className="text-[18px] font-medium"
          style={{ color: data.security.certified ? COLOURS.success.primary : COLOURS.warning.primary }}
        >
          {securityValue}
        </div>
        <div className="text-[11px] text-gray-400 mt-0.5">
          TPA budget: {data.security.tpa_budget ?? '—'} samples
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-1">
          <div className="text-[11px] uppercase tracking-wide text-gray-400">
            Robustness
          </div>
          <InfoTooltip info={METRIC_INFO.robustness} />
        </div>
        <div
          className="text-[18px] font-medium"
          style={{ color: data.robustness.passed ? COLOURS.success.primary : COLOURS.danger.primary }}
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
