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
      <div className="flex justify-end mb-4 animate-fade-in">
        <div className="bg-gray-800 text-white rounded-2xl rounded-br-sm px-4 py-2.5 max-w-[80%]">
          <p className="text-[14px] leading-relaxed">{message.content}</p>
        </div>
      </div>
    );
  }

  const analysis = message.analysis;

  return (
    <div className="mb-5 animate-fade-in">
      <div className="max-w-[90%]">
        <div className="bg-gray-50 border border-gray-100 rounded-2xl rounded-bl-sm px-4 py-3">
          {analysis && (
            <div className="flex items-center gap-1.5 mb-2">
              <span className="text-[10px] font-medium text-gray-400 bg-gray-200/60 px-1.5 py-0.5 rounded">
                {MODEL_DISPLAY_NAMES[analysis.model] ?? analysis.model}
              </span>
            </div>
          )}

          {analysis && (
            <div className="mb-3 pb-3 border-b border-gray-200">
              <MetricCards data={analysis} />
            </div>
          )}

          <div className="text-[14px] leading-relaxed text-gray-700 prose prose-sm max-w-none prose-p:my-1 prose-li:my-0.5 prose-headings:text-gray-800 prose-strong:text-gray-800">
            <Markdown>{message.content}</Markdown>
          </div>

          {analysis && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-center gap-2 text-[12px] font-medium text-gray-600 bg-white border border-gray-200 rounded-lg px-3 py-2 hover:bg-gray-100 hover:border-gray-300 transition-colors"
              >
                <span>
                  {expanded ? 'Hide' : 'View'} claim analysis
                </span>
                <span className="text-gray-400">
                  ({analysis.claims.length} claims)
                </span>
                <svg
                  className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-180' : ''}`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </div>
          )}
        </div>

        {analysis && expanded && (
          <TrustAnalysis data={analysis} />
        )}
      </div>
    </div>
  );
}
