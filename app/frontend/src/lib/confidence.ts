import { COLOURS } from './constants';

export function getConfidenceStyle(score: number) {
  if (score >= 0.80) return {
    bar: COLOURS.success.primary,
    pillBg: COLOURS.success.bg,
    pillText: COLOURS.success.text,
  };
  if (score >= 0.65) return {
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
