import { useRef, useEffect } from 'react';
import { useChat } from './hooks/useChat';
import { ChatMessage } from './components/ChatMessage';
import { ChatInput } from './components/ChatInput';

function App() {
  const { messages, loading, error, send } = useChat();
  const lastMsgRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    lastMsgRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, [messages]);

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white px-6 py-3.5 shrink-0">
        <div className="max-w-3xl mx-auto flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gray-800 flex items-center justify-center">
            <span className="text-white text-[12px] font-bold tracking-tight">TAP</span>
          </div>
          <div>
            <h1 className="text-[15px] font-semibold text-gray-800 leading-tight">
              Trustworthy Answer Protocol
            </h1>
            <p className="text-[11px] text-gray-400">
              Medical LLM with claim-level confidence analysis
            </p>
          </div>
        </div>
      </header>

      {/* Chat thread */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="text-center pt-24">
              <div className="w-12 h-12 rounded-xl bg-gray-800 flex items-center justify-center mx-auto mb-4">
                <span className="text-white text-[16px] font-bold tracking-tight">TAP</span>
              </div>
              <h2 className="text-[18px] font-medium text-gray-700 mb-2">
                Trustworthy Answer Protocol
              </h2>
              <p className="text-[13px] text-gray-400 max-w-md mx-auto mb-6">
                Ask a medical question to see the model's response decomposed into
                individual claims with confidence analysis.
              </p>
              <div className="flex flex-wrap justify-center gap-2 max-w-lg mx-auto">
                {[
                  "What is the recommended treatment for community-acquired pneumonia in an adult with no allergies?",
                  "Explain the differential diagnosis for acute chest pain in a 55-year-old male.",
                  "What are the risks of combining metformin with ACE inhibitors?",
                  "Describe the management of type 2 diabetes in a patient with CKD stage 3.",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    disabled={loading}
                    className="text-[12px] text-gray-500 bg-white border border-gray-200 rounded-lg px-3 py-2 text-left hover:border-gray-300 hover:text-gray-700 transition-colors disabled:opacity-40"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} ref={i === messages.length - 1 ? lastMsgRef : undefined}>
              <ChatMessage message={msg} />
            </div>
          ))}

          {loading && (
            <div className="mb-5 animate-fade-in">
              <div className="max-w-[90%]">
                <div className="bg-gray-50 border border-gray-100 rounded-2xl rounded-bl-sm px-4 py-4">
                  <div className="flex items-center gap-2 text-[13px] text-gray-400">
                    <span className="w-4 h-4 border-2 border-gray-300 border-t-gray-500 rounded-full animate-spin" />
                    Analysing response...
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="text-[13px] text-red-600 bg-red-50 rounded-lg px-3 py-2 mb-4">
              {error}
            </div>
          )}

        </div>
      </div>

      {/* Input */}
      <ChatInput onSubmit={send} loading={loading} />
    </div>
  );
}

export default App;
