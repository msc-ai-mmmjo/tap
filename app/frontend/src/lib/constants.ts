export const MODEL_DISPLAY_NAMES: Record<string, string> = {
  'meta-llama/Meta-Llama-3-8B-Instruct': 'Llama 3 8B',
};

export const CONFIDENCE_THRESHOLDS = {
  high: 0.80,
  moderate: 0.65,
} as const;

export const COLOURS = {
  success: {
    primary: '#1D9E75',
    bg: '#E1F5EE',
    text: '#085041',
  },
  warning: {
    primary: '#EF9F27',
    bg: '#FAEEDA',
    text: '#633806',
  },
  danger: {
    primary: '#E24B4A',
    bg: '#FCEBEB',
    text: '#791F1F',
  },
} as const;
