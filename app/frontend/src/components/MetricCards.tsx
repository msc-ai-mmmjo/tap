import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { AnalysisResponse } from '../types/api';

interface Props {
  data: AnalysisResponse;
}

interface MetricInfo {
  definition: string;
  paper: string;
}

const METRIC_INFO: Record<'certainty' | 'security' | 'robustness', MetricInfo> = {
  certainty: {
    definition:
      "How likely each claim in the answer is to be factually correct. Scores near 100% mean high confidence; low scores flag claims worth double-checking before acting on them. Currently available for MCQ questions only; a dash is shown when no estimate is produced.",
    paper:
      'Method: Kapoor et al., "Large Language Models Must Be Taught to Know What They Don\'t Know" (NeurIPS 2024).',
  },
  security: {
    definition:
      "Each token is cross-checked by several independent heads in the model. Whenever they disagree with the draft, the token is rewritten. Fewer rewrites means stronger agreement across heads, and a more trustworthy answer.",
    paper:
      'Method: Ghitu & Wicker, "Towards Poisoning Robustness Certification for Natural Language Generation".',
  },
  robustness: {
    definition:
      "Whether the answer holds up against jailbreak attempts — short gibberish strings appended to the prompt that try to flip the response. 'Passed' means no attempt succeeded.",
    paper:
      'Method: Kumar et al., "AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts".',
  },
};

function InfoTooltip({ info, label }: { info: MetricInfo; label: string }) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const id = useId();

  const place = () => {
    const t = triggerRef.current?.getBoundingClientRect();
    const tip = tooltipRef.current?.getBoundingClientRect();
    if (!t) return;
    const width = tip?.width ?? 280;
    const height = tip?.height ?? 80;
    const left = Math.min(
      window.innerWidth - width - 12,
      Math.max(12, t.left + t.width / 2 - width / 2),
    );
    const overflowsBelow = t.bottom + 10 + height > window.innerHeight;
    const top = overflowsBelow ? t.top - height - 10 : t.bottom + 10;
    setPos({ top, left });
  };

  useLayoutEffect(() => {
    if (open) place();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    const onScroll = () => place();
    window.addEventListener('keydown', onKey);
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [open]);

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={`About the ${label} metric`}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center justify-center w-4 h-4 rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--color-accent)]"
        style={{
          color: open ? 'var(--color-accent)' : 'var(--color-ink-muted)',
        }}
      >
        <svg
          aria-hidden
          className="w-3.5 h-3.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.75}
        >
          <circle cx="12" cy="12" r="10" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 16v-4M12 8h.01" />
        </svg>
      </button>
      {open &&
        createPortal(
          <div
            ref={tooltipRef}
            id={id}
            role="tooltip"
            style={{
              top: pos?.top ?? -9999,
              left: pos?.left ?? -9999,
              width: 280,
              background: 'var(--color-ink)',
              color: 'var(--color-paper)',
            }}
            className="fixed px-3.5 py-3 text-[12px] leading-[1.55] shadow-xl pointer-events-none z-50 animate-tooltip-in"
          >
            <div className="mb-1.5">{info.definition}</div>
            <div
              className="font-mono text-[10px] uppercase tracking-wider pt-1.5 mt-1.5"
              style={{
                color: 'var(--color-paper-2)',
                borderTop: '1px solid rgba(255,255,255,0.18)',
              }}
            >
              {info.paper}
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}

interface CellProps {
  index: string;
  label: string;
  info: MetricInfo;
  value: string;
  valueColour?: string;
  caption: string;
  isFirst?: boolean;
}

function MetricCell({ index, label, info, value, valueColour, caption, isFirst }: CellProps) {
  return (
    <div
      className="px-5 py-4"
      style={{
        borderLeft: isFirst ? 'none' : '1px solid var(--color-rule)',
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <div
          className="font-mono text-[10px] uppercase tracking-[0.16em] flex items-baseline gap-1.5"
          style={{ color: 'var(--color-ink-muted)' }}
        >
          <span style={{ color: 'var(--color-accent)' }}>{index}</span>
          {label}
        </div>
        <InfoTooltip info={info} label={label} />
      </div>
      <div
        className="font-mono text-[24px] tabular-nums leading-none"
        style={{ color: valueColour ?? 'var(--color-ink)' }}
      >
        {value}
      </div>
      <div
        className="text-[11.5px] leading-snug mt-2"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        {caption}
      </div>
    </div>
  );
}

export function MetricCards({ data }: Props) {
  const { certified, resampled, tokens } = data.security;
  const resampledCount = resampled.length;
  const totalTokens = tokens.length;

  let securityValue: string;
  let securityColour: string;
  let securityCaption: string;
  if (certified === null) {
    securityValue = '—';
    securityColour = 'var(--color-ink-muted)';
    securityCaption = 'Fallback: no PoE guarantee';
  } else {
    securityValue = `${resampledCount} swap${resampledCount !== 1 ? 's' : ''}`;
    securityColour =
      resampledCount === 0
        ? 'var(--color-ok)'
        : resampledCount <= 3
          ? 'var(--color-warn)'
          : 'var(--color-bad)';
    securityCaption = `${resampledCount} of ${totalTokens} tokens resampled during verification`;
  }

  const { overall: certaintyOverall } = data.uncertainty;
  const certaintyValue =
    certaintyOverall === null ? '—' : `${Math.round(certaintyOverall * 100)}%`;
  const certaintyColour =
    certaintyOverall === null ? 'var(--color-ink-muted)' : undefined;
  const certaintyCaption =
    certaintyOverall === null
      ? 'Fallback: no uncertainty estimate'
      : 'Model-estimated probability this answer is correct';

  const robustnessValue = data.robustness.passed ? 'Passed' : 'Failed';

  return (
    <div
      className="grid grid-cols-3"
      style={{
        background: 'var(--color-card)',
        border: '1px solid var(--color-rule)',
      }}
    >
      <MetricCell
        isFirst
        index="01"
        label="Certainty"
        info={METRIC_INFO.certainty}
        value={certaintyValue}
        valueColour={certaintyColour}
        caption={certaintyCaption}
      />
      <MetricCell
        index="02"
        label="Security"
        info={METRIC_INFO.security}
        value={securityValue}
        valueColour={securityColour}
        caption={securityCaption}
      />
      <MetricCell
        index="03"
        label="Robustness"
        info={METRIC_INFO.robustness}
        value={robustnessValue}
        valueColour={data.robustness.passed ? 'var(--color-ok)' : 'var(--color-bad)'}
        caption={data.robustness.detail}
      />
    </div>
  );
}
