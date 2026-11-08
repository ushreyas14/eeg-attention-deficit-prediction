"""
Feature Extraction
EEG: Band power, theta/alpha ratio, frontal alpha asymmetry,
     spectral entropy, sample entropy, alpha coherence
Behavioral: RT, accuracy from symbol search & sequence learning tasks
Labels: attention score + binary High/Low from participants.tsv
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch
import antropy as ant
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
BIDS_ROOT     = os.path.join(os.path.dirname(__file__), 'data', 'raw')

BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

# CBCL clinical cutoff: T-score >= 65 → High attention problems
ATTENTION_CUTOFF = 65.0


# ── EEG features ─────────────────────────────────────────────────────────────

def compute_band_powers(psd: np.ndarray, freqs: np.ndarray) -> dict:
    features = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        power = np.mean(psd[:, idx], axis=1)   # [n_channels]
        for ch, val in enumerate(power):
            features[f'{band}_ch{ch}'] = float(val)
    return features


def compute_theta_alpha_ratio(psd: np.ndarray, freqs: np.ndarray, n_channels: int) -> dict:
    features = {}
    theta_idx = np.where((freqs >= 4) & (freqs < 8))[0]
    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]
    theta = np.mean(psd[:, theta_idx], axis=1)
    alpha = np.mean(psd[:, alpha_idx], axis=1)
    for ch in range(n_channels):
        features[f'theta_alpha_ratio_ch{ch}'] = float(theta[ch] / (alpha[ch] + 1e-10))
    # Mean ratio across all channels (single global biomarker)
    features['theta_alpha_ratio_mean'] = float(np.mean(theta / (alpha + 1e-10)))
    return features


def compute_frontal_alpha_asymmetry(psd: np.ndarray, freqs: np.ndarray,
                                     ch_names: list) -> dict:
    features = {}
    alpha_idx = np.where((freqs >= 8) & (freqs < 13))[0]
    alpha_power = np.mean(psd[:, alpha_idx], axis=1)

    pairs = [('F3', 'F4'), ('F7', 'F8'), ('Fp1', 'Fp2')]
    for left, right in pairs:
        if left in ch_names and right in ch_names:
            l_idx = ch_names.index(left)
            r_idx = ch_names.index(right)
            asym = (np.log(alpha_power[r_idx] + 1e-10)
                    - np.log(alpha_power[l_idx] + 1e-10))
            features[f'frontal_alpha_asym_{left}_{right}'] = float(asym)
    return features


def compute_spectral_entropy(epochs_data: np.ndarray, sfreq: float) -> dict:
    features = {}
    n_channels = epochs_data.shape[1]
    # Average across epochs for a stable estimate
    mean_signal = epochs_data.mean(axis=0)   # [channels, times]
    for ch in range(n_channels):
        se = ant.spectral_entropy(mean_signal[ch], sf=sfreq,
                                   method='welch', normalize=True)
        features[f'spectral_entropy_ch{ch}'] = float(se)
    return features


def compute_sample_entropy(epochs_data: np.ndarray) -> dict:
    features = {}
    n_channels = epochs_data.shape[1]
    mean_signal = epochs_data.mean(axis=0)
    for ch in range(n_channels):
        try:
            se = ant.sample_entropy(mean_signal[ch])
        except Exception:
            se = np.nan
        features[f'sample_entropy_ch{ch}'] = float(se)
    return features


def compute_alpha_coherence(epochs: mne.Epochs) -> dict:
    """Pairwise alpha coherence using manual computation (no extra dependency)."""
    features = {}
    data = epochs.get_data()          # [epochs, channels, times]
    sfreq = epochs.info['sfreq']
    n_ch = data.shape[1]
    alpha_lo, alpha_hi = 8, 13

    # Compute per-epoch FFT and extract alpha band
    freqs = np.fft.rfftfreq(data.shape[2], d=1.0 / sfreq)
    alpha_mask = (freqs >= alpha_lo) & (freqs <= alpha_hi)

    fft_data = np.fft.rfft(data, axis=2)   # [epochs, channels, freqs]
    alpha_fft = fft_data[:, :, alpha_mask]  # [epochs, channels, alpha_freqs]

    # Coherence between pairs (upper triangle, limit to keep feature count manageable)
    pairs_done = 0
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            if pairs_done >= 50:          # cap at 50 pairs to avoid feature explosion
                break
            xi = alpha_fft[:, i, :]
            xj = alpha_fft[:, j, :]
            cross = np.mean(xi * np.conj(xj), axis=0)
            auto_i = np.mean(np.abs(xi) ** 2, axis=0)
            auto_j = np.mean(np.abs(xj) ** 2, axis=0)
            coh = np.abs(cross) / (np.sqrt(auto_i * auto_j) + 1e-10)
            features[f'alpha_coh_{i}_{j}'] = float(np.mean(coh))
            pairs_done += 1
        if pairs_done >= 50:
            break
    return features


def extract_eeg_features(epochs: mne.Epochs) -> dict:
    data = epochs.get_data()                # [epochs, channels, times]
    sfreq = epochs.info['sfreq']
    ch_names = epochs.ch_names
    n_ch = data.shape[1]

    # PSD averaged across epochs
    freqs, psd = welch(data, fs=sfreq, axis=-1)   # [epochs, ch, freqs]
    psd_mean = psd.mean(axis=0)                    # [ch, freqs]

    features = {}
    features.update(compute_band_powers(psd_mean, freqs))
    features.update(compute_theta_alpha_ratio(psd_mean, freqs, n_ch))
    features.update(compute_frontal_alpha_asymmetry(psd_mean, freqs, ch_names))
    features.update(compute_spectral_entropy(data, sfreq))
    features.update(compute_sample_entropy(data))
    features.update(compute_alpha_coherence(epochs))
    return features


# ── Behavioral features ───────────────────────────────────────────────────────

def extract_behavioral_features(subject_id: str) -> dict:
    """Extract RT and accuracy from symbol search and sequence learning tasks."""
    features = {}
    tasks = {
        'symbolSearch':     ['symbolSearch'],
        'seqLearning':      ['seqLearning6target', 'seqLearning8target'],
    }

    for task_group, task_names in tasks.items():
        rts, accs = [], []
        for task in task_names:
            events_path = os.path.join(
                BIDS_ROOT, f'sub-{subject_id}', 'eeg',
                f'sub-{subject_id}_task-{task}_events.tsv'
            )
            if not os.path.exists(events_path):
                continue
            df = pd.read_csv(events_path, sep='\t')
            if 'response_time' in df.columns:
                rt_vals = pd.to_numeric(df['response_time'], errors='coerce').dropna()
                rts.extend(rt_vals.tolist())
            if 'accuracy' in df.columns:
                acc_vals = pd.to_numeric(df['accuracy'], errors='coerce').dropna()
                accs.extend(acc_vals.tolist())

        features[f'{task_group}_mean_rt']      = float(np.mean(rts))   if rts  else np.nan
        features[f'{task_group}_std_rt']       = float(np.std(rts))    if rts  else np.nan
        features[f'{task_group}_accuracy']     = float(np.mean(accs))  if accs else np.nan
        features[f'{task_group}_rt_iqr']       = float(np.percentile(rts, 75) - np.percentile(rts, 25)) if len(rts) > 3 else np.nan

    return features


# ── Labels ────────────────────────────────────────────────────────────────────

def load_labels() -> pd.DataFrame:
    participants_file = os.path.join(BIDS_ROOT, 'participants.tsv')
    df = pd.read_csv(participants_file, sep='\t')
    df['participant_id'] = df['participant_id'].str.replace('sub-', '')
    df = df.dropna(subset=['attention'])
    # Binary label: 1 = High attention problems (clinical threshold T >= 65)
    df['attention_high'] = (df['attention'] >= ATTENTION_CUTOFF).astype(int)
    return df[['participant_id', 'attention', 'attention_high',
               'p_factor', 'internalizing', 'externalizing', 'age', 'sex']]


# ── Main extraction loop ──────────────────────────────────────────────────────

def build_feature_matrix(max_subjects: int = None) -> pd.DataFrame:
    labels_df = load_labels()
    subjects = labels_df['participant_id'].tolist()
    if max_subjects:
        subjects = subjects[:max_subjects]

    rows = []
    for subj in subjects:
        epoch_path = os.path.join(PROCESSED_DIR, f'sub-{subj}_epochs.fif')
        if not os.path.exists(epoch_path):
            print(f"  Skipping {subj} — no preprocessed epochs found")
            continue

        print(f"  Extracting features: {subj}")
        try:
            epochs = mne.read_epochs(epoch_path, verbose=False, preload=True)
            eeg_feats = extract_eeg_features(epochs)
            beh_feats = extract_behavioral_features(subj)

            row = {'subject_id': subj}
            row.update(eeg_feats)
            row.update(beh_feats)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR {subj}: {e}")

    if not rows:
        raise RuntimeError("No features extracted. Run 1_preprocess.py first.")

    features_df = pd.DataFrame(rows)
    merged = features_df.merge(labels_df, left_on='subject_id',
                                right_on='participant_id', how='inner')
    merged.drop(columns=['participant_id'], inplace=True)

    out_path = os.path.join(PROCESSED_DIR, 'feature_matrix.csv')
    merged.to_csv(out_path, index=False)
    print(f"\nFeature matrix saved → {out_path}")
    print(f"Shape: {merged.shape}  |  "
          f"High attention: {merged['attention_high'].sum()}  |  "
          f"Low attention: {(merged['attention_high'] == 0).sum()}")
    return merged


if __name__ == '__main__':
    build_feature_matrix()
