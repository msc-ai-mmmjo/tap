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
    text: '#0e3d2b',
  },
  warning: {
    primary: '#a05a14',
    text: '#5a330a',
  },
  danger: {
    primary: '#b91c1c',
    text: '#5a1414',
  },
} as const;
