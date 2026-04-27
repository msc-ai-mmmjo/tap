import type { SecurityResample } from '../types/api';

// Thresholds on the 0–8 validity radius scale (higher = more decisive = lower risk)
export const SECURITY_RISK_THRESHOLDS = { low: 4, moderate: 2 } as const;

/**
 * Returns the mean validity radius of resampled tokens, or null when the
 * field is absent (older backend). Higher mean = swaps were more decisive =
 * lower poisoning risk.
 */
export function meanValidityRadius(resampled: SecurityResample[]): number | null {
  const withRadius = resampled.filter((r) => r.validity_radius != null);
  if (withRadius.length === 0) return null;
  return withRadius.reduce((sum, r) => sum + r.validity_radius!, 0) / withRadius.length;
}

export function computeSecurityRisk(
  resampled: SecurityResample[],
): 'low' | 'moderate' | 'high' | null {
  const mean = meanValidityRadius(resampled);
  if (mean === null) return null;
  if (mean >= SECURITY_RISK_THRESHOLDS.low) return 'low';
  if (mean >= SECURITY_RISK_THRESHOLDS.moderate) return 'moderate';
  return 'high';
}
