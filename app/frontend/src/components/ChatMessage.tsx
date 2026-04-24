import { useState } from 'react';
import Markdown from 'react-markdown';
import type { ChatMessage as ChatMessageType } from '../types/api';
import { MetricCards } from './MetricCards';
import { TrustAnalysis } from './TrustAnalysis';
import { MODEL_DISPLAY_NAMES } from '../lib/constants';

interface Props {
  message: ChatMessageType;
}

export function ChatMessage({ message }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (message.role === 'user') {
    return (
      <div className="mb-8 animate-fade-in">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-2"
          style={{ color: 'var(--color-accent-warm)' }}
        >
          — Re:
        </div>
        <p
          className="text-[14px] leading-snug"
          style={{ color: 'var(--color-ink-2)' }}
        >
          {message.content}
        </p>
      </div>
    );
  }

  const analysis = message.analysis;

  return (
    <div className="mb-10 animate-fade-in">
      <div
        className="flex items-center justify-between mb-3"
        style={{ borderTop: '1px solid var(--color-rule)', paddingTop: 8 }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em]"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          — Response
        </div>
        {analysis && (
          <div
            className="font-mono text-[10px] uppercase tracking-[0.16em]"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            Model · {MODEL_DISPLAY_NAMES[analysis.model] ?? analysis.model}
          </div>
        )}
      </div>

      <div
        className="text-[17px] leading-[1.75] prose max-w-none prose-p:my-3 prose-li:my-1.5 prose-headings:font-display prose-headings:font-normal prose-strong:font-semibold"
        style={{ color: 'var(--color-ink-2)' }}
      >
        <Markdown>{message.content}</Markdown>
      </div>

      {analysis && (
        <div className="mt-6">
          <MetricCards data={analysis} />
        </div>
      )}

      {analysis && (
        <div className="mt-6">
          <button
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
            className="font-mono text-[10.5px] uppercase tracking-[0.16em] inline-flex items-center gap-2 py-1.5 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
            style={{ color: 'var(--color-ink-muted)' }}
          >
            <span
              aria-hidden
              className="inline-block w-3 text-center leading-none"
            >
              {expanded ? '−' : '+'}
            </span>
            {expanded ? 'Hide' : 'Inspect'} details
            {analysis.security.resampled.length > 0 && (
              <span style={{ color: 'var(--color-warn)' }}>
                · {analysis.security.resampled.length} swap{analysis.security.resampled.length !== 1 ? 's' : ''}
              </span>
            )}
          </button>
        </div>
      )}

      {analysis && expanded && <TrustAnalysis data={analysis} />}
    </div>
  );
}
