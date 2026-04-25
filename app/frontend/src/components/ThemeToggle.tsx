import { useTheme } from '../hooks/useTheme';

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isDark = theme === 'dark';

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle dark mode"
      aria-pressed={isDark}
      className="font-mono text-[10px] uppercase tracking-[0.18em] cursor-pointer focus:outline-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
      style={{
        color: 'var(--color-ink-muted)',
        outlineColor: 'var(--color-accent)',
        padding: '6px 4px',
        minHeight: 24,
      }}
    >
      <span style={{ opacity: isDark ? 0.4 : 1 }}>LIGHT</span>
      <span style={{ opacity: 0.4, margin: '0 6px' }}>·</span>
      <span style={{ opacity: isDark ? 1 : 0.4 }}>DARK</span>
    </button>
  );
}
