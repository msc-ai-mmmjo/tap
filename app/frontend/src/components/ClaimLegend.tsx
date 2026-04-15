import type { CSSProperties } from 'react';

const moderateUnderline: CSSProperties = {
  textDecorationLine: 'underline',
  textDecorationStyle: 'dashed',
  textDecorationColor: 'var(--color-warn)',
  textDecorationThickness: '1.5px',
  textUnderlineOffset: '4px',
};

const lowUnderline: CSSProperties = {
  textDecorationLine: 'underline',
  textDecorationStyle: 'solid',
  textDecorationColor: 'var(--color-bad)',
  textDecorationThickness: '2px',
  textUnderlineOffset: '4px',
};

function LegendRow({
  sample,
  sampleStyle,
  label,
  caption,
}: {
  sample: string;
  sampleStyle: CSSProperties;
  label: string;
  caption: string;
}) {
  return (
    <div className="flex items-baseline gap-2.5 min-w-0">
      <span
        className="text-[13px] shrink-0"
        style={{ color: 'var(--color-ink-2)', ...sampleStyle }}
      >
        {sample}
      </span>
      <span
        className="font-mono text-[10px] uppercase tracking-[0.14em] truncate"
        style={{ color: 'var(--color-ink-soft)' }}
      >
        <span style={{ color: 'var(--color-ink-muted)' }}>{label}</span>
        <span> · {caption}</span>
      </span>
    </div>
  );
}

export function ClaimLegend() {
  return (
    <div
      className="px-5 py-3 flex flex-wrap items-baseline gap-x-6 gap-y-2"
      style={{
        background: 'var(--color-card)',
        borderLeft: '1px solid var(--color-rule)',
        borderRight: '1px solid var(--color-rule)',
        borderBottom: '1px solid var(--color-rule)',
      }}
    >
      <span
        className="font-mono text-[10px] uppercase tracking-[0.18em]"
        style={{ color: 'var(--color-ink-muted)' }}
      >
        — Claim annotations
      </span>
      <LegendRow
        sample="claim text"
        sampleStyle={lowUnderline}
        label="Low"
        caption="Cross-check with authoritative source"
      />
      <LegendRow
        sample="claim text"
        sampleStyle={moderateUnderline}
        label="Moderate"
        caption="Verify with clinical reference"
      />
      <LegendRow
        sample="claim text"
        sampleStyle={{}}
        label="High"
        caption="No flag raised"
      />
    </div>
  );
}
