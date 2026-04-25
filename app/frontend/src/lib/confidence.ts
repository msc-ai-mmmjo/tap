import { COLOURS, CONFIDENCE_THRESHOLDS } from './constants';

export function getConfidenceStyle(score: number) {
  if (score >= CONFIDENCE_THRESHOLDS.high) return {
    bar: COLOURS.success,
    pillText: COLOURS.success,
    tierLabel: 'High confidence',
  };
  if (score >= CONFIDENCE_THRESHOLDS.moderate) return {
    bar: COLOURS.warning,
    pillText: COLOURS.warning,
    tierLabel: 'Moderate',
  };
  return {
    bar: COLOURS.danger,
    pillText: COLOURS.danger,
    tierLabel: 'Low confidence',
  };
}
