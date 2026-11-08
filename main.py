"""
Main Pipeline Runner
Predicting Attention Deficits in Children Using EEG and Behavioral Data
Dataset: OpenNeuro ds005507 (HBN-EEG)

Usage:
  python main.py --mode simulate   # run on synthetic data (no download needed)
  python main.py --mode real       # run on actual ds005507 data
  python main.py --mode eval_only  # skip training, just re-generate plots
"""

import argparse
import os
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║   Predicting Attention Deficits in Children                      ║
║   Using EEG and Behavioral Data                                  ║
║   Dataset: OpenNeuro ds005507 (HBN-EEG)                         ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def run_simulate_mode():
    print("\n[MODE: SIMULATE] Generating synthetic dataset...")
    from simulate_data import simulate
    simulate()


def run_preprocess():
    print("\n[STEP 1/4] EEG Preprocessing...")
    from preprocess import run_preprocessing
    run_preprocessing()


def run_feature_extraction():
    print("\n[STEP 2/4] Feature Extraction (EEG + Behavioral)...")
    from extract_features import build_feature_matrix
    build_feature_matrix()


def run_training():
    print("\n[STEP 3/4] Model Training...")
    from train_model import run_training as _train
    clf_results, reg_results, imp_df = _train()
    return clf_results, reg_results


def run_evaluation():
    print("\n[STEP 4/4] Evaluation & Visualization...")
    from evaluate import run_evaluation as _eval
    _eval()


def print_final_summary():
    import pandas as pd
    clf_path = os.path.join(RESULTS_DIR, 'models', 'classification_results.csv')
    reg_path = os.path.join(RESULTS_DIR, 'models', 'regression_results.csv')

    print("\n" + "="*65)
    print(" FINAL RESULTS SUMMARY")
    print("="*65)

    if os.path.exists(clf_path):
        clf_df = pd.read_csv(clf_path)
        best = clf_df.iloc[0]
        print(f"\n Best Classifier : {best['Model']}")
        print(f"  Accuracy        : {best['Accuracy']:.4f} ± {best['Accuracy_std']:.4f}")
        print(f"  F1 Score        : {best['F1']:.4f} ± {best['F1_std']:.4f}")
        print(f"  AUC-ROC         : {best['AUC_ROC']:.4f} ± {best['AUC_ROC_std']:.4f}")
        print(f"  Cohen's Kappa   : {best['Cohen_Kappa']:.4f}")

    if os.path.exists(reg_path):
        reg_df = pd.read_csv(reg_path)
        best_r = reg_df.iloc[0]
        print(f"\n Best Regressor  : {best_r['Model']}")
        print(f"  MAE             : {best_r['MAE']:.4f} ± {best_r['MAE_std']:.4f}")
        print(f"  R²              : {best_r['R2']:.4f}")
        print(f"  Pearson r       : {best_r['Pearson_r']:.4f}")

    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    if os.path.exists(plots_dir):
        plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        print(f"\n Plots saved ({len(plots)}):")
        for p in sorted(plots):
            print(f"   results/plots/{p}")

    print("\n" + "="*65)


def main():
    banner()
    parser = argparse.ArgumentParser(
        description='Attention Deficit Prediction from EEG and Behavioral Data'
    )
    parser.add_argument(
        '--mode', choices=['simulate', 'real', 'eval_only'],
        default='simulate',
        help=(
            'simulate: use synthetic data (default)\n'
            'real: use downloaded ds005507 dataset\n'
            'eval_only: regenerate plots from existing feature_matrix.csv'
        )
    )
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Limit number of subjects (for real mode)')
    args = parser.parse_args()

    start = time.time()

    if args.mode == 'simulate':
        run_simulate_mode()
        run_training()
        run_evaluation()

    elif args.mode == 'real':
        run_preprocess()
        run_feature_extraction()
        run_training()
        run_evaluation()

    elif args.mode == 'eval_only':
        run_evaluation()

    print_final_summary()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == '__main__':
    # Fix relative imports when running as script
    sys.path.insert(0, os.path.dirname(__file__))

    # Rename modules for import (strip number prefix)
    import importlib.util, types

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    base = os.path.dirname(__file__)
    load_module('preprocess',      os.path.join(base, '1_preprocess.py'))
    load_module('extract_features', os.path.join(base, '2_extract_features.py'))
    load_module('train_model',     os.path.join(base, '3_train_model.py'))
    load_module('evaluate',        os.path.join(base, '4_evaluate.py'))
    load_module('simulate_data',   os.path.join(base, 'simulate_data.py'))

    main()
