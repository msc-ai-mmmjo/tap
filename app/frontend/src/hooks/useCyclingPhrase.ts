import { useEffect, useState } from 'react';

/**
 * Cycles through a list of phrases at a fixed interval while active.
 * Resets to the first phrase when active flips from true to false.
 */
export function useCyclingPhrase(
  phrases: readonly string[],
  intervalMs: number,
  active: boolean,
): string {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setIndex(0);
      return;
    }
    const id = setInterval(() => {
      setIndex((i) => (i + 1) % phrases.length);
    }, intervalMs);
    return () => clearInterval(id);
  }, [active, intervalMs, phrases.length]);

  return phrases[index] ?? phrases[0] ?? '';
}
