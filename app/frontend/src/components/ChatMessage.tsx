import Markdown from 'react-markdown';
import type { ChatMessage as ChatMessageType } from '../types/api';
import { MetricCards } from './MetricCards';
import { ClaimLegend } from './ClaimLegend';
import { InlineClaims } from './InlineClaims';
import { MODEL_DISPLAY_NAMES } from '../lib/constants';

interface Props {
  message: ChatMessageType;
}

export function ChatMessage({ message }: Props) {
  if (message.role === 'user') {
    return (
      <div className="mb-8 animate-fade-in">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-2"
          style={{ color: 'var(--color-accent)' }}
        >
          — Query
        </div>
        <p
          className="font-display text-[22px] leading-[1.25]"
          style={{ color: 'var(--color-ink)' }}
        >
          “{message.content}”
        </p>
      </div>
    );
  }

  const analysis = message.analysis;

  return (
    <div className="mb-10 animate-fade-in">
      <div
        className="flex items-center justify-between mb-3"
        style={{ borderTop: '1px solid var(--color-ink)', paddingTop: 8 }}
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

      {analysis && (
        <div className="mb-5">
          <MetricCards data={analysis} />
          <ClaimLegend />
        </div>
      )}

      {analysis ? (
        <InlineClaims text={analysis.raw_response} claims={analysis.claims} />
      ) : (
        <div
          className="text-[15px] leading-[1.7] prose max-w-none prose-p:my-2.5 prose-li:my-1 prose-headings:font-display prose-headings:font-normal prose-strong:font-semibold"
          style={{ color: 'var(--color-ink-2)' }}
        >
          <Markdown>{message.content}</Markdown>
        </div>
      )}
    </div>
  );
}
