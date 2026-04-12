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
    <form onSubmit={handleSubmit} className="border-t border-gray-200 bg-white p-4">
      <div className="max-w-3xl mx-auto flex gap-3 items-end">
        <textarea
          className="flex-1 border border-gray-200 rounded-xl px-4 py-3 text-[14px] text-gray-800 resize-none focus:outline-none focus:ring-2 focus:ring-gray-200 placeholder:text-gray-400"
          rows={2}
          placeholder="Enter a medical query..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="px-5 py-3 text-[13px] font-medium text-white bg-gray-800 rounded-xl hover:bg-gray-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analysing
            </span>
          ) : (
            'Send'
          )}
        </button>
      </div>
    </form>
  );
}
