import { COLOURS, CONFIDENCE_THRESHOLDS } from './constants';

export function getConfidenceStyle(score: number) {
  if (score >= CONFIDENCE_THRESHOLDS.high) return {
    bar: COLOURS.success.primary,
    pillBg: COLOURS.success.bg,
    pillText: COLOURS.success.text,
  };
  if (score >= CONFIDENCE_THRESHOLDS.moderate) return {
    bar: COLOURS.warning.primary,
    pillBg: COLOURS.warning.bg,
    pillText: COLOURS.warning.text,
  };
  return {
    bar: COLOURS.danger.primary,
    pillBg: COLOURS.danger.bg,
    pillText: COLOURS.danger.text,
  };
}
