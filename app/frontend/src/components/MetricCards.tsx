import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { AnalysisResponse } from '../types/api';
import { CONFIDENCE_THRESHOLDS, ROBUSTNESS_FLIP_WARN_RATIO } from '../lib/constants';

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
      "How likely the answer is to be correct. For multiple-choice questions the model reports a probability directly. For free-form answers we resample several responses and measure how much they agree. Scores near 100% mean high confidence; low scores flag answers worth double-checking before acting on them.",
    paper:
      'Methods: Kapoor et al., "Large Language Models Must Be Taught to Know What They Don\'t Know" (NeurIPS 2024); Nikitin et al., "Kernel Language Entropy" (NeurIPS 2024).',
  },
  security: {
    definition:
      "Defends against training-data tampering (poisoning). Each token is cross-checked by several independent heads in the model; whenever they disagree with the draft, the token is rewritten. Fewer rewrites means stronger agreement across heads, and a more trustworthy answer.",
    paper:
      'Method: Ghitu & Wicker, "Towards Poisoning Robustness Certification for Natural Language Generation".',
  },
  robustness: {
    definition:
      "How the answer holds up against jailbreak attempts. We run several short attack strings and count how many times the answer changed meaning (letter for MCQ, semantic for NLP). Fewer flips means a more robust answer.",
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
                borderTop: '1px solid var(--color-rule-on-ink)',
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
  severityLabel?: string;
  caption: string;
}

function MetricCell({ index, label, info, value, valueColour, severityLabel, caption }: CellProps) {
  return (
    <div className="px-5 py-4 border-l border-rule first:border-l-0">
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
      <div className="flex items-baseline gap-2 leading-none">
        <span
          className="font-mono text-[18px] tabular-nums"
          style={{ color: valueColour ?? 'var(--color-ink)' }}
        >
          {value}
        </span>
        {severityLabel && (
          <span
            className="font-mono text-[9.5px] uppercase tracking-[0.14em]"
            style={{ color: valueColour ?? 'var(--color-ink-soft)' }}
          >
            {severityLabel}
          </span>
        )}
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

type Severity = 'ok' | 'warn' | 'bad' | 'na';

const SEVERITY_COLOUR: Record<Severity, string> = {
  ok: 'var(--color-ok)',
  warn: 'var(--color-warn)',
  bad: 'var(--color-bad)',
  na: 'var(--color-ink-muted)',
};

const SEVERITY_LABEL: Record<Severity, string | undefined> = {
  ok: 'Low risk',
  warn: 'Moderate',
  bad: 'High risk',
  na: undefined,
};

export function MetricCards({ data }: Props) {
  const { overall: certaintyOverall } = data.uncertainty;
  const certaintyValue =
    certaintyOverall === null ? '—' : `${Math.round(certaintyOverall * 100)}%`;
  let certaintySeverity: Severity;
  let certaintyCaption: string;
  if (certaintyOverall === null) {
    certaintySeverity = 'na';
    certaintyCaption = 'Fallback: no uncertainty estimate';
  } else if (certaintyOverall >= CONFIDENCE_THRESHOLDS.high) {
    certaintySeverity = 'ok';
    certaintyCaption = 'High confidence this answer is correct';
  } else if (certaintyOverall >= CONFIDENCE_THRESHOLDS.moderate) {
    certaintySeverity = 'warn';
    certaintyCaption = 'Moderate confidence, verify key claims';
  } else {
    certaintySeverity = 'bad';
    certaintyCaption = 'Low confidence, double-check before acting';
  }

  const { certified, resampled, tokens } = data.security;
  const resampledCount = resampled.length;
  const totalTokens = tokens.length;

  let securityValue: string;
  let securitySeverity: Severity;
  let securityCaption: string;
  if (certified === null) {
    securityValue = '—';
    securitySeverity = 'na';
    securityCaption = 'Fallback: no PoE guarantee';
  } else {
    securityValue = `${resampledCount} swap${resampledCount !== 1 ? 's' : ''}`;
    if (resampledCount === 0) {
      securitySeverity = 'ok';
      securityCaption = `All ${totalTokens} tokens agreed across heads`;
    } else if (resampledCount <= 3) {
      securitySeverity = 'warn';
      securityCaption = `${resampledCount} of ${totalTokens} tokens resampled, minor disagreement`;
    } else {
      securitySeverity = 'bad';
      securityCaption = `${resampledCount} of ${totalTokens} tokens resampled, heavy disagreement`;
    }
  }

  const robustness = data.robustness;
  let robustnessValue: string;
  let robustnessSeverity: Severity;
  let robustnessCaption: string;

  if (robustness.type === 'unavailable') {
    robustnessValue = '—';
    robustnessSeverity = 'na';
    robustnessCaption = 'Fallback: no adversarial examples';
  } else {
    const stable = robustness.flipped === 0;
    const flipRatio =
      robustness.attempts > 0 ? robustness.flipped / robustness.attempts : 0;
    robustnessValue = stable
      ? 'Stable'
      : `${robustness.flipped}/${robustness.attempts} flipped`;
    if (stable) {
      robustnessSeverity = 'ok';
      robustnessCaption =
        robustness.type === 'nlp'
          ? `Meaning held across ${robustness.attempts} attacks`
          : `Answer held across ${robustness.attempts} attacks`;
    } else if (flipRatio <= ROBUSTNESS_FLIP_WARN_RATIO) {
      robustnessSeverity = 'warn';
      robustnessCaption =
        robustness.type === 'nlp'
          ? 'Some attacks shifted the meaning'
          : 'Some attacks flipped the letter';
    } else {
      robustnessSeverity = 'bad';
      robustnessCaption =
        robustness.type === 'nlp'
          ? 'Many attacks shifted the meaning'
          : 'Many attacks flipped the letter';
    }
  }

  return (
    <div
      className="grid grid-cols-3"
      style={{
        background: 'var(--color-card)',
        border: '1px solid var(--color-rule-soft)',
      }}
    >
      <MetricCell
        index="01"
        label="Certainty"
        info={METRIC_INFO.certainty}
        value={certaintyValue}
        valueColour={SEVERITY_COLOUR[certaintySeverity]}
        severityLabel={SEVERITY_LABEL[certaintySeverity]}
        caption={certaintyCaption}
      />
      <MetricCell
        index="02"
        label="Security"
        info={METRIC_INFO.security}
        value={securityValue}
        valueColour={SEVERITY_COLOUR[securitySeverity]}
        severityLabel={SEVERITY_LABEL[securitySeverity]}
        caption={securityCaption}
      />
      <MetricCell
        index="03"
        label="Robustness"
        info={METRIC_INFO.robustness}
        value={robustnessValue}
        valueColour={SEVERITY_COLOUR[robustnessSeverity]}
        severityLabel={SEVERITY_LABEL[robustnessSeverity]}
        caption={robustnessCaption}
      />
    </div>
  );
}
