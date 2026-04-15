import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';

export interface HoverCardTriggerProps {
  ref: (el: HTMLElement | null) => void;
  'aria-describedby': string | undefined;
  'aria-expanded': boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onFocus: () => void;
  onBlur: () => void;
  onClick: () => void;
}

interface Props {
  content: ReactNode;
  width?: number;
  children: (props: HoverCardTriggerProps) => ReactNode;
}

export function HoverCard({ content, width = 280, children }: Props) {
  const triggerRef = useRef<HTMLElement | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const tipId = useId();

  const place = useCallback(() => {
    const a = triggerRef.current?.getBoundingClientRect();
    const tip = tipRef.current?.getBoundingClientRect();
    if (!a) return;
    const height = tip?.height ?? 80;
    const left = Math.min(
      window.innerWidth - width - 12,
      Math.max(12, a.left + a.width / 2 - width / 2),
    );
    const overflowsBelow = a.bottom + 10 + height > window.innerHeight;
    const top = overflowsBelow ? a.top - height - 10 : a.bottom + 10;
    setPos({ top, left });
  }, [width]);

  useLayoutEffect(() => {
    if (open) place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.blur();
      }
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('scroll', place, true);
    window.addEventListener('resize', place);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', place, true);
      window.removeEventListener('resize', place);
    };
  }, [open, place]);

  const attachTrigger = useCallback((el: HTMLElement | null) => {
    triggerRef.current = el;
  }, []);

  const triggerProps: HoverCardTriggerProps = {
    ref: attachTrigger,
    'aria-describedby': open ? tipId : undefined,
    'aria-expanded': open,
    onMouseEnter: () => setOpen(true),
    onMouseLeave: () => setOpen(false),
    onFocus: () => setOpen(true),
    onBlur: () => setOpen(false),
    onClick: () => setOpen((v) => !v),
  };

  return (
    <>
      {/* triggerProps.ref is a callback ref, not a current read — safe at render time */}
      {/* eslint-disable-next-line react-hooks/refs */}
      {children(triggerProps)}
      {open &&
        createPortal(
          <div
            ref={tipRef}
            id={tipId}
            role="tooltip"
            style={{
              top: pos?.top ?? -9999,
              left: pos?.left ?? -9999,
              width,
              background: 'var(--color-ink)',
              color: 'var(--color-paper)',
            }}
            className="fixed px-3.5 py-3 text-[12px] leading-[1.55] shadow-xl pointer-events-none z-50 animate-tooltip-in"
          >
            {content}
          </div>,
          document.body,
        )}
    </>
  );
}
