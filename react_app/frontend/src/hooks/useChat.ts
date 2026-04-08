import { useState } from 'react';
import type { ChatMessage, AnalysisResponse } from '../types/api';

const API_BASE = 'http://localhost:8000';

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send(content: string) {
    const userMsg: ChatMessage = { role: 'user', content };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/analyse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: updated.map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const analysis: AnalysisResponse = await res.json();
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: analysis.raw_response,
        analysis,
      };

      setMessages([...updated, assistantMsg]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  return { messages, loading, error, send };
}
