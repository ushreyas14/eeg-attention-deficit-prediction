"""
EEG Preprocessing Pipeline
Dataset: OpenNeuro ds005507 (HBN-EEG)
Task: RestingState
"""

import os
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
from autoreject import AutoReject
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

BIDS_ROOT = os.path.join(os.path.dirname(__file__), 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

EPOCH_DURATION = 2.0       # seconds per epoch
L_FREQ = 1.0               # high-pass cutoff
H_FREQ = 45.0              # low-pass cutoff
NOTCH_FREQ = 60.0          # line noise (US)
REJECT_THRESH = 150e-6     # 150 µV epoch rejection threshold
N_ICA_COMPONENTS = 20


def load_raw(subject_id: str) -> mne.io.Raw:
    bids_path = BIDSPath(
        subject=subject_id,
        task='RestingState',
        datatype='eeg',
        root=BIDS_ROOT
    )
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()
    raw.pick_types(eeg=True, stim=False, exclude='bads')
    return raw


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    return raw


def rereference(raw: mne.io.Raw) -> mne.io.Raw:
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    return raw


def run_ica(raw: mne.io.Raw) -> mne.io.Raw:
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    n_comp = min(N_ICA_COMPONENTS, n_eeg - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_comp,
        random_state=42,
        max_iter='auto'
    )
    ica.fit(raw)

    try:
        muscle_indices, _ = ica.find_bads_muscle(raw)
        ica.exclude = list(set(muscle_indices))
    except Exception:
        pass

    raw = ica.apply(raw)
    return raw


def epoch_data(raw: mne.io.Raw) -> mne.Epochs:
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_DURATION,
        preload=True,
        verbose=False
    )
    # Basic amplitude-based rejection
    epochs.drop_bad(reject={'eeg': REJECT_THRESH})
    return epochs


def autoreject_epochs(epochs: mne.Epochs) -> mne.Epochs:
    ar = AutoReject(random_state=42, verbose=False)
    epochs_clean, _ = ar.fit_transform(epochs, return_log=True)
    return epochs_clean


def preprocess_subject(subject_id: str, use_autoreject: bool = False) -> mne.Epochs | None:
    try:
        print(f"  Loading {subject_id}...")
        raw = load_raw(subject_id)

        print(f"  Filtering {subject_id}...")
        raw = filter_raw(raw)
        raw = rereference(raw)

        print(f"  ICA {subject_id}...")
        raw = run_ica(raw)

        print(f"  Epoching {subject_id}...")
        epochs = epoch_data(raw)

        if use_autoreject and len(epochs) > 10:
            epochs = autoreject_epochs(epochs)

        # Save epochs
        out_path = os.path.join(PROCESSED_DIR, f'sub-{subject_id}_epochs.fif')
        epochs.save(out_path, overwrite=True)
        print(f"  Saved {subject_id} → {len(epochs)} clean epochs")
        return epochs

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        return None


def get_subject_list() -> list:
    participants_file = os.path.join(BIDS_ROOT, 'participants.tsv')
    if not os.path.exists(participants_file):
        raise FileNotFoundError(
            f"participants.tsv not found at {participants_file}\n"
            "Download ds005507 from OpenNeuro and place it in data/raw/"
        )
    df = pd.read_csv(participants_file, sep='\t')
    # Keep only subjects with attention score
    df = df.dropna(subset=['attention'])
    # Strip 'sub-' prefix if present
    subjects = df['participant_id'].str.replace('sub-', '').tolist()
    return subjects


def run_preprocessing(max_subjects: int = None, use_autoreject: bool = False):
    subjects = get_subject_list()
    if max_subjects:
        subjects = subjects[:max_subjects]

    print(f"\nPreprocessing {len(subjects)} subjects...\n")
    success, failed = 0, 0

    for subj in subjects:
        result = preprocess_subject(subj, use_autoreject=use_autoreject)
        if result is not None:
            success += 1
        else:
            failed += 1

    print(f"\nDone. Success: {success} | Failed: {failed}")


if __name__ == '__main__':
    run_preprocessing(max_subjects=50)
