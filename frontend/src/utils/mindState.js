const MIND_STATE_METADATA = {
  relaxation: {
    label: 'Relaxation',
    description: 'Alpha balance & HRV stability.',
  },
  engagement: {
    label: 'Engagement',
    description: 'Goal-directed focus vs baseline.',
  },
  stress: {
    label: 'Stress / Scatter',
    description: 'Sympathetic activation and restlessness.',
  },
};

export function clampScore(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(-1, Math.min(1, value));
}

export function normalizedDeviation(value) {
  return clampScore((value ?? 0) / 3);
}

function describeScore(score, key) {
  if (key === 'stress') {
    if (score >= 0.45) return 'High tension';
    if (score >= 0.15) return 'Alert';
    if (score <= -0.45) return 'Deeply calm';
    if (score <= -0.15) return 'Settled';
    return 'Balanced';
  }
  if (score >= 0.45) return 'Elevated';
  if (score >= 0.15) return 'Steady';
  if (score <= -0.45) return 'Low';
  if (score <= -0.15) return 'Subdued';
  return 'Neutral';
}

export function computeMindStateScores(deviations = {}) {
  const normalized = (key) => normalizedDeviation(deviations?.[key]);
  const relaxation = clampScore(
    (normalized('alpha_relaxation') + normalized('theta_relaxation') + normalized('hrv_rmssd') - normalized('hrv_lf_hf')) / 4
  );
  const engagement = clampScore(
    (normalized('beta_concentration') - 0.6 * normalized('theta_relaxation') + 0.3 * normalized('alpha_relaxation')) / 1.9
  );
  const stress = clampScore(
    (normalized('hrv_lf_hf') - normalized('hrv_rmssd') + 0.5 * normalized('beta_concentration')) / 2.5
  );
  const hrvComposite = clampScore(
    (normalized('hrv_rmssd') + normalized('hrv_sdnn') - normalized('hrv_lf_hf')) / 3
  );
  return { relaxation, engagement, stress, hrvComposite };
}

export function deriveMindStates(deviations = {}) {
  const scores = computeMindStateScores(deviations);
  return ['relaxation', 'engagement', 'stress'].map((key) => {
    const meta = MIND_STATE_METADATA[key] || {};
    const score = scores[key];
    const percent = Math.round((score + 1) * 50);
    return {
      key,
      score,
      percent,
      label: meta.label || key,
      description: meta.description || '',
      status: describeScore(score, key),
    };
  });
}

export { MIND_STATE_METADATA };
