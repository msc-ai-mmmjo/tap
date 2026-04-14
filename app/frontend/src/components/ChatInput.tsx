import { useState } from 'react';

interface Props {
  onSubmit: (message: string) => void;
  loading: boolean;
}

export function ChatInput({ onSubmit, loading }: Props) {
  const [value, setValue] = useState('');

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim() && !loading) {
      onSubmit(value.trim());
      setValue('');
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="shrink-0"
      style={{
        borderTop: '1px solid var(--color-rule)',
        background: 'var(--color-paper)',
      }}
    >
      <div className="max-w-3xl mx-auto px-6 py-4">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-2 flex items-center gap-2"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span style={{ color: 'var(--color-accent)' }}>▸</span>
          Enter clinical query
        </div>
        <div
          className="flex gap-3 items-end p-2"
          style={{
            background: 'var(--color-card)',
            border: '1px solid var(--color-rule)',
          }}
        >
          <textarea
            className="flex-1 bg-transparent resize-none focus:outline-none px-2 py-1.5 text-[14.5px] leading-[1.5] placeholder:opacity-60"
            style={{ color: 'var(--color-ink)' }}
            rows={2}
            placeholder="e.g. Treatment for community-acquired pneumonia, no allergies…"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            aria-label="Clinical query"
          />
          <button
            type="submit"
            disabled={loading || !value.trim()}
            className="font-mono text-[11px] uppercase tracking-[0.16em] px-4 py-2.5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
            style={{
              background: 'var(--color-ink)',
              color: 'var(--color-paper)',
            }}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span
                  className="w-2 h-2 rounded-full animate-pulse"
                  style={{ background: 'var(--color-paper)' }}
                />
                Analysing
              </span>
            ) : (
              <span className="flex items-center gap-2">
                Send
                <span aria-hidden>↵</span>
              </span>
            )}
          </button>
        </div>
      </div>
    </form>
  );
}
