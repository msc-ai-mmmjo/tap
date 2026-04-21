export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama 3 8B',
  Hydra: 'Hydra',
};

export const CONFIDENCE_THRESHOLDS = {
  high: 0.80,
  moderate: 0.65,
} as const;

export const COLOURS = {
  success: {
    primary: '#1f6f4f',
    bg: '#d6e8de',
    text: '#0e3d2b',
  },
  warning: {
    primary: '#a05a14',
    bg: '#f0e0c4',
    text: '#5a330a',
  },
  danger: {
    primary: '#9b2a2a',
    bg: '#efd5d5',
    text: '#5a1414',
  },
} as const;

export type NlpRobustnessScore = 0 | 0.5 | 1 | 1.5 | 2;

export const ROBUSTNESS_NLP_LABELS: Record<NlpRobustnessScore, string> = {
  0: 'Divergent',
  0.5: 'Mostly divergent',
  1: 'Ambiguous',
  1.5: 'Mostly equivalent',
  2: 'Equivalent',
};

export function getRobustnessNlpStyle(score: NlpRobustnessScore) {
  if (score >= 1.5) {
    return {
      bar: COLOURS.success.primary,
      pillBg: COLOURS.success.bg,
      pillText: COLOURS.success.text,
    };
  }
  if (score >= 1) {
    return {
      bar: COLOURS.warning.primary,
      pillBg: COLOURS.warning.bg,
      pillText: COLOURS.warning.text,
    };
  }
  return {
    bar: COLOURS.danger.primary,
    pillBg: COLOURS.danger.bg,
    pillText: COLOURS.danger.text,
  };
}
