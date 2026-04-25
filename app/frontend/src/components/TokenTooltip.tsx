import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

interface Props {
  token: string;
  tooltipBody: ReactNode;
  triggerStyle?: CSSProperties;
}

export function TokenTooltip({ token, tooltipBody, triggerStyle }: Props) {
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
        style={{ cursor: 'help', ...triggerStyle }}
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
            {tooltipBody}
          </div>,
          document.body,
        )}
    </>
  );
}
