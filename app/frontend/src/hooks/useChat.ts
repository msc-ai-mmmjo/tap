import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, AnalysisResponse } from '../types/api';

const API_BASE = import.meta.env.VITE_API_BASE;
if (!API_BASE) throw new Error('VITE_API_BASE is not set');

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  async function send(content: string) {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

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
        signal: controller.signal,
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
      if (e instanceof DOMException && e.name === 'AbortError') return;
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        abortRef.current = null;
      }
    }
  }

  return { messages, loading, error, send };
}
