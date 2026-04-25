import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';

function readAttribute(): Theme {
  if (typeof document === 'undefined') return 'light';
  const value = document.documentElement.getAttribute('data-theme');
  return value === 'dark' ? 'dark' : 'light';
}

function writeAttribute(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme);
}

function persist(theme: Theme): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* localStorage unavailable (private mode) — ignore */
  }
}

export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(readAttribute);

  useEffect(() => {
    writeAttribute(theme);
    persist(theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }, []);

  return { theme, toggle };
}
