"""
Model Training
Task 1: Binary Classification — Low vs High Attention (T-score >= 65)
Task 2: Regression — Predict continuous attention score
Models: Logistic Regression, SVM, Random Forest, XGBoost, Gradient Boosting
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, cohen_kappa_score,
    mean_absolute_error, r2_score
)
import xgboost as xgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
MODELS_DIR    = os.path.join(os.path.dirname(__file__), 'results', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

LABEL_COL_CLASS = 'attention_high'       # binary
LABEL_COL_REG   = 'attention'            # continuous
META_COLS = ['subject_id', 'attention', 'attention_high',
             'p_factor', 'internalizing', 'externalizing', 'age', 'sex']
N_FOLDS = 5
RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = os.path.join(PROCESSED_DIR, 'feature_matrix.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            "feature_matrix.csv not found. Run 2_extract_features.py first.\n"
            "Or run simulate_data.py to generate mock data."
        )
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y_class = df[LABEL_COL_CLASS].values.astype(int)
    y_reg   = df[LABEL_COL_REG].values.astype(float)
    return df, X, y_class, y_reg, feature_cols


def make_preprocessor():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])


def build_classifiers() -> dict:
    pre = make_preprocessor()
    return {
        'Logistic Regression': Pipeline([
            ('pre', make_preprocessor()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        'SVM': Pipeline([
            ('pre', make_preprocessor()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
        ]),
        'Random Forest': Pipeline([
            ('pre', make_preprocessor()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        'Gradient Boosting': Pipeline([
            ('pre', make_preprocessor()),
            ('clf', GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        'XGBoost': Pipeline([
            ('pre', make_preprocessor()),
            ('clf', xgb.XGBClassifier(n_estimators=200, use_label_encoder=False,
                                       eval_metric='logloss', random_state=RANDOM_STATE,
                                       verbosity=0))
        ]),
    }


def build_regressors() -> dict:
    return {
        'Ridge': Pipeline([
            ('pre', make_preprocessor()),
            ('reg', Ridge(alpha=1.0))
        ]),
        'ElasticNet': Pipeline([
            ('pre', make_preprocessor()),
            ('reg', ElasticNet(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        'SVR': Pipeline([
            ('pre', make_preprocessor()),
            ('reg', SVR(kernel='rbf'))
        ]),
        'Random Forest': Pipeline([
            ('pre', make_preprocessor()),
            ('reg', RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE))
        ]),
        'XGBoost': Pipeline([
            ('pre', make_preprocessor()),
            ('reg', xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE,
                                      verbosity=0))
        ]),
    }


# ── Classification ─────────────────────────────────────────────────────────────

def train_classification(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    print("\n" + "="*60)
    print("TASK 1: BINARY CLASSIFICATION (Low vs High Attention)")
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    classifiers = build_classifiers()
    results = []

    for name, pipeline in classifiers.items():
        cv_results = cross_validate(
            pipeline, X, y, cv=skf,
            scoring=['accuracy', 'f1', 'roc_auc'],
            return_train_score=False
        )
        # Cohen's kappa via manual fold loop
        kappas = []
        for train_idx, test_idx in skf.split(X, y):
            pipeline.fit(X[train_idx], y[train_idx])
            y_pred = pipeline.predict(X[test_idx])
            kappas.append(cohen_kappa_score(y[test_idx], y_pred))

        row = {
            'Model': name,
            'Accuracy':  round(cv_results['test_accuracy'].mean(), 4),
            'Accuracy_std': round(cv_results['test_accuracy'].std(), 4),
            'F1':        round(cv_results['test_f1'].mean(), 4),
            'F1_std':    round(cv_results['test_f1'].std(), 4),
            'AUC_ROC':   round(cv_results['test_roc_auc'].mean(), 4),
            'AUC_ROC_std': round(cv_results['test_roc_auc'].std(), 4),
            'Cohen_Kappa': round(np.mean(kappas), 4),
        }
        results.append(row)
        print(f"  {name:25s} | Acc: {row['Accuracy']:.4f} ± {row['Accuracy_std']:.4f} "
              f"| F1: {row['F1']:.4f} | AUC: {row['AUC_ROC']:.4f} "
              f"| Kappa: {row['Cohen_Kappa']:.4f}")

    results_df = pd.DataFrame(results).sort_values('AUC_ROC', ascending=False)
    results_df.to_csv(os.path.join(MODELS_DIR, 'classification_results.csv'), index=False)

    # Save best model
    best_name = results_df.iloc[0]['Model']
    best_pipeline = build_classifiers()[best_name]
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_classifier.pkl'))
    print(f"\n  Best classifier: {best_name} → saved")

    return results_df


# ── Regression ────────────────────────────────────────────────────────────────

def train_regression(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    print("\n" + "="*60)
    print("TASK 2: REGRESSION (Predict Attention Score)")
    print("="*60)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    regressors = build_regressors()
    results = []

    for name, pipeline in regressors.items():
        maes, r2s, corrs = [], [], []
        for train_idx, test_idx in kf.split(X):
            pipeline.fit(X[train_idx], y[train_idx])
            y_pred = pipeline.predict(X[test_idx])
            maes.append(mean_absolute_error(y[test_idx], y_pred))
            r2s.append(r2_score(y[test_idx], y_pred))
            corr = np.corrcoef(y[test_idx], y_pred)[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0.0)

        row = {
            'Model': name,
            'MAE':      round(np.mean(maes), 4),
            'MAE_std':  round(np.std(maes), 4),
            'R2':       round(np.mean(r2s), 4),
            'R2_std':   round(np.std(r2s), 4),
            'Pearson_r': round(np.mean(corrs), 4),
        }
        results.append(row)
        print(f"  {name:25s} | MAE: {row['MAE']:.4f} ± {row['MAE_std']:.4f} "
              f"| R²: {row['R2']:.4f} | r: {row['Pearson_r']:.4f}")

    results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
    results_df.to_csv(os.path.join(MODELS_DIR, 'regression_results.csv'), index=False)

    # Save best model
    best_name = results_df.iloc[0]['Model']
    best_pipeline = build_regressors()[best_name]
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_regressor.pkl'))
    print(f"\n  Best regressor: {best_name} → saved")

    return results_df


# ── Feature importance ────────────────────────────────────────────────────────

def save_feature_importance(X: np.ndarray, y_class: np.ndarray,
                             feature_cols: list):
    """Train RF on full data and save feature importances."""
    pre = make_preprocessor()
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    X_proc = pre.fit_transform(X)
    rf.fit(X_proc, y_class)

    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    imp_df.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)
    print(f"\n  Top 10 features:")
    print(imp_df.head(10).to_string(index=False))
    return imp_df


# ── Entry point ───────────────────────────────────────────────────────────────

def run_training():
    df, X, y_class, y_reg, feature_cols = load_features()
    print(f"\nDataset: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"High attention: {y_class.sum()} | Low attention: {(y_class == 0).sum()}")

    clf_results = train_classification(X, y_class)
    reg_results = train_regression(X, y_reg)
    imp_df = save_feature_importance(X, y_class, feature_cols)

    return clf_results, reg_results, imp_df


if __name__ == '__main__':
    run_training()
