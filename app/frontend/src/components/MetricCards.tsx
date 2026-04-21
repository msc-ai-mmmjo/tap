import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { AnalysisResponse, RobustnessStatus } from '../types/api';
import { getConfidenceStyle } from '../lib/confidence';
import {
  ROBUSTNESS_NLP_LABELS,
  getRobustnessNlpStyle,
  type NlpRobustnessScore,
} from '../lib/constants';

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
      "How likely the response is to be correct overall. For multiple-choice queries this comes from a calibrated uncertainty head; for open-ended queries it comes from Kernel Language Entropy over resampled responses.",
    paper:
      'Methods: Kapoor et al., "Large Language Models Must Be Taught to Know What They Don\'t Know" (NeurIPS 2024); Nikitin et al., "Kernel Language Entropy" (2024).',
  },
  security: {
    definition:
      "During PoE speculative verification, tokens where the draft head disagrees with the rest are resampled from the product-of-experts distribution. Resampled tokens are highlighted; a higher swap count suggests the draft head had weaker consensus with its peers.",
    paper:
      'Method: Product-of-Experts speculative verification; softer variant of Wicker et al. stability radius.',
  },
  robustness: {
    definition:
      "Compares the response to the same prompt extended with an AmpleGCG adversarial suffix. MCQ: binary flip of the chosen letter. NLP: bidirectional NLI agreement between clean and attacked responses, bucketed into five levels.",
    paper:
      'Methods: Kumar et al., "AmpleGCG-Plus" (2024); bidirectional NLI scoring.',
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

function robustnessCellProps(robustness: RobustnessStatus): {
  value: string;
  valueColour: string;
  caption: string;
} {
  if (robustness.type === 'mcq') {
    return {
      value: robustness.flipped ? 'Flipped' : 'Stable',
      valueColour: robustness.flipped ? 'var(--color-bad)' : 'var(--color-ok)',
      caption: `${robustness.original_choice} → ${robustness.attacked_choice}`,
    };
  }
  const score = robustness.bidirectional_score as NlpRobustnessScore;
  const style = getRobustnessNlpStyle(score);
  return {
    value: ROBUSTNESS_NLP_LABELS[score],
    valueColour: style.pillText,
    caption: 'Bidirectional NLI agreement vs adversarial suffix',
  };
}

export function MetricCards({ data }: Props) {
  const certaintyPct = Math.round(data.uncertainty.overall * 100);
  const certaintyStyle = getConfidenceStyle(data.uncertainty.overall);

  const securityCount = data.security.resampled.length;
  const securityColour = securityCount === 0 ? 'var(--color-ok)' : 'var(--color-warn)';

  const rob = robustnessCellProps(data.robustness);

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
        value={`${certaintyPct}%`}
        valueColour={certaintyStyle.pillText}
        caption="Likelihood response is correct"
      />
      <MetricCell
        index="02"
        label="Security"
        info={METRIC_INFO.security}
        value={`${securityCount} swap${securityCount === 1 ? '' : 's'}`}
        valueColour={securityColour}
        caption={`Out of ${data.security.tokens.length} tokens generated`}
      />
      <MetricCell
        index="03"
        label="Robustness"
        info={METRIC_INFO.robustness}
        value={rob.value}
        valueColour={rob.valueColour}
        caption={rob.caption}
      />
    </div>
  );
}
