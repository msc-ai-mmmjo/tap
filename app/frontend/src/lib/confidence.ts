import { COLOURS, CONFIDENCE_THRESHOLDS } from './constants';

export function getConfidenceStyle(score: number) {
  if (score >= CONFIDENCE_THRESHOLDS.high) return {
    bar: COLOURS.success.primary,
    pillText: COLOURS.success.text,
    tierLabel: 'High confidence',
  };
  if (score >= CONFIDENCE_THRESHOLDS.moderate) return {
    bar: COLOURS.warning.primary,
    pillText: COLOURS.warning.text,
    tierLabel: 'Moderate',
  };
  return {
    bar: COLOURS.danger.primary,
    pillText: COLOURS.danger.text,
    tierLabel: 'Low confidence',
  };
}
