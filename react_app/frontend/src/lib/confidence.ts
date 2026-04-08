export function getConfidenceStyle(score: number) {
  if (score >= 0.80) return {
    bar: '#1D9E75',
    pillBg: '#E1F5EE',
    pillText: '#085041',
  };
  if (score >= 0.65) return {
    bar: '#EF9F27',
    pillBg: '#FAEEDA',
    pillText: '#633806',
  };
  return {
    bar: '#E24B4A',
    pillBg: '#FCEBEB',
    pillText: '#791F1F',
  };
}
