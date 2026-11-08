"""
Synthetic Data Generator
Simulates the feature_matrix.csv that 1_preprocess.py + 2_extract_features.py would produce.
Use this to test the full pipeline WITHOUT downloading ds005507.

EEG patterns are grounded in real ADHD neuroscience:
  - High attention group: elevated theta/alpha ratio, reduced alpha power,
    lower spectral entropy, slower RTs, lower task accuracy
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SUBJECTS   = 180       # realistic for one HBN-EEG release
N_CHANNELS   = 19        # standard 10-20 system


def simulate(n_subjects: int = N_SUBJECTS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    # ── Ground truth attention scores ────────────────────────────────────────
    # ~30% high attention problems (T >= 65), ~70% low
    attention_scores = np.concatenate([
        rng.normal(loc=52, scale=8,  size=int(n_subjects * 0.70)),   # Low group
        rng.normal(loc=72, scale=7,  size=int(n_subjects * 0.30)),   # High group
    ])
    attention_scores = np.clip(attention_scores, 30, 100)
    rng.shuffle(attention_scores)

    high_mask = (attention_scores >= 65).astype(float)

    rows = []
    for i in range(n_subjects):
        is_high = high_mask[i]
        row = {'subject_id': f'sub-{i+1:04d}'}

        # ── Band powers per channel ───────────────────────────────────────────
        for ch in range(N_CHANNELS):
            # Theta elevated in high-attention group
            row[f'delta_ch{ch}'] = rng.uniform(0.8, 2.5) * (1.0 + 0.1 * is_high)
            row[f'theta_ch{ch}'] = rng.uniform(1.0, 3.5) * (1.0 + 0.35 * is_high) + rng.normal(0, 0.2)
            # Alpha reduced in high-attention group
            row[f'alpha_ch{ch}'] = rng.uniform(1.5, 4.0) * (1.0 - 0.25 * is_high) + rng.normal(0, 0.2)
            row[f'beta_ch{ch}']  = rng.uniform(0.5, 2.0) * (1.0 - 0.15 * is_high)
            row[f'gamma_ch{ch}'] = rng.uniform(0.1, 0.8)

        # ── Theta/alpha ratio ─────────────────────────────────────────────────
        for ch in range(N_CHANNELS):
            theta = row[f'theta_ch{ch}']
            alpha = row[f'alpha_ch{ch}']
            row[f'theta_alpha_ratio_ch{ch}'] = theta / (alpha + 1e-10)
        row['theta_alpha_ratio_mean'] = np.mean(
            [row[f'theta_alpha_ratio_ch{ch}'] for ch in range(N_CHANNELS)]
        )

        # ── Frontal alpha asymmetry ───────────────────────────────────────────
        row['frontal_alpha_asym_F3_F4']   = rng.normal(-0.1 * is_high, 0.15)
        row['frontal_alpha_asym_F7_F8']   = rng.normal(-0.08 * is_high, 0.12)
        row['frontal_alpha_asym_Fp1_Fp2'] = rng.normal(-0.05 * is_high, 0.10)

        # ── Spectral entropy ──────────────────────────────────────────────────
        for ch in range(N_CHANNELS):
            row[f'spectral_entropy_ch{ch}'] = rng.uniform(0.6, 0.95) * (1.0 - 0.08 * is_high)

        # ── Sample entropy ────────────────────────────────────────────────────
        for ch in range(N_CHANNELS):
            row[f'sample_entropy_ch{ch}'] = rng.uniform(0.5, 2.0) * (1.0 - 0.12 * is_high)

        # ── Alpha coherence (50 pairs) ────────────────────────────────────────
        pair_idx = 0
        for ci in range(N_CHANNELS):
            for cj in range(ci + 1, N_CHANNELS):
                if pair_idx >= 50:
                    break
                row[f'alpha_coh_{ci}_{cj}'] = rng.uniform(0.1, 0.8) * (1.0 - 0.10 * is_high)
                pair_idx += 1
            if pair_idx >= 50:
                break

        # ── Behavioral features ───────────────────────────────────────────────
        # Slower RT, lower accuracy in high-attention group
        row['symbolSearch_mean_rt']  = rng.normal(650 + 120 * is_high, 80)
        row['symbolSearch_std_rt']   = rng.normal(150 + 50 * is_high, 30)
        row['symbolSearch_accuracy'] = np.clip(rng.normal(0.78 - 0.15 * is_high, 0.08), 0, 1)
        row['symbolSearch_rt_iqr']   = rng.normal(180 + 60 * is_high, 40)

        row['seqLearning_mean_rt']   = rng.normal(700 + 100 * is_high, 90)
        row['seqLearning_std_rt']    = rng.normal(160 + 55 * is_high, 35)
        row['seqLearning_accuracy']  = np.clip(rng.normal(0.72 - 0.14 * is_high, 0.09), 0, 1)
        row['seqLearning_rt_iqr']    = rng.normal(200 + 70 * is_high, 45)

        # ── Labels & demographics ─────────────────────────────────────────────
        row['attention']      = float(attention_scores[i])
        row['attention_high'] = int(is_high)
        row['p_factor']       = float(rng.normal(0.3 * is_high, 0.4))
        row['internalizing']  = float(rng.normal(50 + 10 * is_high, 8))
        row['externalizing']  = float(rng.normal(48 + 12 * is_high, 9))
        row['age']            = float(rng.randint(5, 22))
        row['sex']            = rng.choice(['M', 'F'])

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DIR, 'feature_matrix.csv')
    df.to_csv(out_path, index=False)

    print(f"Simulated feature matrix saved → {out_path}")
    print(f"Shape: {df.shape}")
    print(f"High attention (T≥65): {df['attention_high'].sum()} subjects")
    print(f"Low attention  (T<65): {(df['attention_high'] == 0).sum()} subjects")
    print(f"Features: {df.shape[1] - 8} EEG + behavioral columns")
    return df


if __name__ == '__main__':
    simulate()
