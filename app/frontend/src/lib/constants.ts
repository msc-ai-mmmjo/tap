export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama 3 8B',
};

export const CONFIDENCE_THRESHOLDS = {
  high: 0.80,
  moderate: 0.65,
} as const;

export const ROBUSTNESS_FLIP_WARN_RATIO = 1 / 3;

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
