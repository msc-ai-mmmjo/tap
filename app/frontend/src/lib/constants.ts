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
    primary: 'var(--color-ok)',
    text: 'var(--color-ok)',
  },
  warning: {
    primary: 'var(--color-warn)',
    text: 'var(--color-warn)',
  },
  danger: {
    primary: 'var(--color-bad)',
    text: 'var(--color-bad)',
  },
} as const;
