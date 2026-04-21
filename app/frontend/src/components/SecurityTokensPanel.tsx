import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { SecurityStatus, SecurityResample } from '../types/api';

interface Props {
  data: SecurityStatus;
}

function ResampledToken({ token, resample }: { token: string; resample: SecurityResample }) {
  const triggerRef = useRef<HTMLSpanElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const id = useId();

  const place = () => {
    const t = triggerRef.current?.getBoundingClientRect();
    const tip = tooltipRef.current?.getBoundingClientRect();
    if (!t) return;
    const width = tip?.width ?? 180;
    const height = tip?.height ?? 36;
    const left = Math.min(
      window.innerWidth - width - 12,
      Math.max(12, t.left + t.width / 2 - width / 2),
    );
    const overflowsBelow = t.bottom + 8 + height > window.innerHeight;
    const top = overflowsBelow ? t.top - height - 8 : t.bottom + 8;
    setPos({ top, left });
  };

  useLayoutEffect(() => {
    if (open) place();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onScroll = () => place();
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [open]);

  return (
    <>
      <span
        ref={triggerRef}
        tabIndex={0}
        role="button"
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        style={{
          color: 'var(--color-accent)',
          borderBottom: '1px dotted var(--color-accent)',
          textDecoration: 'underline',
          textDecorationStyle: 'solid',
          textUnderlineOffset: '3px',
          cursor: 'help',
        }}
      >
        {token}
      </span>
      {open &&
        pos &&
        createPortal(
          <div
            ref={tooltipRef}
            id={id}
            role="tooltip"
            style={{
              top: pos.top,
              left: pos.left,
              background: 'var(--color-ink)',
              color: 'var(--color-paper)',
            }}
            className="fixed px-2.5 py-1.5 text-[11.5px] font-mono shadow-xl pointer-events-none z-50"
          >
            {resample.old_token} → {resample.new_token}
          </div>,
          document.body,
        )}
    </>
  );
}

export function SecurityTokensPanel({ data }: Props) {
  const resampleByIndex = new Map<number, SecurityResample>(
    data.resampled.map((r) => [r.index, r]),
  );

  return (
    <div className="mt-5 animate-fade-in">
      <div
        className="px-5 pt-4 pb-4"
        style={{
          background: 'var(--color-card)',
          border: '1px solid var(--color-rule)',
        }}
      >
        <div
          className="font-mono text-[10px] uppercase tracking-[0.18em] mb-3 flex items-center justify-between"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span>— Security token stream</span>
          <span style={{ color: 'var(--color-ink-soft)' }}>
            {data.resampled.length} / {data.tokens.length} resampled
          </span>
        </div>
        {data.resampled.length === 0 && (
          <div
            className="text-[11.5px] italic mb-2"
            style={{ color: 'var(--color-ink-soft)' }}
          >
            No resampled tokens.
          </div>
        )}
        <div
          className="font-mono text-[13px] leading-[1.7]"
          style={{ color: 'var(--color-ink-soft)' }}
        >
          {data.tokens.map((tok, i) => {
            const r = resampleByIndex.get(i);
            return (
              <span key={i}>
                {r ? <ResampledToken token={tok} resample={r} /> : tok}
                {i < data.tokens.length - 1 ? ' ' : ''}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}
