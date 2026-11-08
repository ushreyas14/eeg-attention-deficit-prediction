"""
Evaluation & Visualization
Plots: ROC curve, confusion matrix, regression scatter, feature importance,
       SHAP summary, learning curve, model comparison bar chart,
       attention score distribution, correlation heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, KFold, learning_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    mean_absolute_error, r2_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')
MODELS_DIR    = os.path.join(os.path.dirname(__file__), 'results', 'models')
PLOTS_DIR     = os.path.join(os.path.dirname(__file__), 'results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

META_COLS = ['subject_id', 'attention', 'attention_high',
             'p_factor', 'internalizing', 'externalizing', 'age', 'sex']

PALETTE = {'Low': '#2196F3', 'High': '#F44336'}
FIG_DPI = 150
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': FIG_DPI,
})


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data():
    path = os.path.join(PROCESSED_DIR, 'feature_matrix.csv')
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values
    y_class = df['attention_high'].values.astype(int)
    y_reg   = df['attention'].values.astype(float)
    return df, X, y_class, y_reg, feature_cols


def preprocess_X(X):
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    return scl.fit_transform(imp.fit_transform(X))


# ── 1. Attention score distribution ──────────────────────────────────────────

def plot_attention_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax = axes[0]
    low = df[df['attention_high'] == 0]['attention']
    high = df[df['attention_high'] == 1]['attention']
    ax.hist(low,  bins=20, alpha=0.7, color=PALETTE['Low'],  label='Low (T<65)')
    ax.hist(high, bins=20, alpha=0.7, color=PALETTE['High'], label='High (T≥65)')
    ax.axvline(65, color='black', linestyle='--', linewidth=1.5, label='Clinical cutoff (T=65)')
    ax.set_xlabel('CBCL Attention T-score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Attention Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()

    # Class balance pie
    ax2 = axes[1]
    counts = [len(low), len(high)]
    labels = [f'Low Attention\n(n={len(low)})', f'High Attention\n(n={len(high)})']
    colors = [PALETTE['Low'], PALETTE['High']]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    ax2.set_title('Class Balance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '1_attention_distribution.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 2. ROC Curves (all classifiers) ──────────────────────────────────────────

def plot_roc_curves(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost':             xgb.XGBClassifier(n_estimators=200, eval_metric='logloss',
                                                   random_state=42, verbosity=0),
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_proc = preprocess_X(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')

    for (name, clf), color in zip(classifiers.items(), colors):
        tprs, aucs_list = [], []
        mean_fpr = np.linspace(0, 1, 100)
        for train_idx, test_idx in skf.split(X_proc, y):
            clf.fit(X_proc[train_idx], y[train_idx])
            proba = clf.predict_proba(X_proc[test_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y[test_idx], proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs_list.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs_list)
        std_auc  = np.std(aucs_list)
        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                label=f'{name} (AUC={mean_auc:.3f} ± {std_auc:.3f})')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Classifiers (5-Fold CV)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '2_roc_curves.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 3. Confusion matrix (best classifier) ────────────────────────────────────

def plot_confusion_matrix(X: np.ndarray, y: np.ndarray):
    model_path = os.path.join(MODELS_DIR, 'best_classifier.pkl')
    if not os.path.exists(model_path):
        print("  Skipping confusion matrix — best_classifier.pkl not found")
        return

    clf = joblib.load(model_path)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(clf.predict(X[test_idx]))

    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Low Attention', 'High Attention']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix (5-Fold CV)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '3_confusion_matrix.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 4. Regression scatter (predicted vs actual) ───────────────────────────────

def plot_regression_scatter(X: np.ndarray, y: np.ndarray):
    model_path = os.path.join(MODELS_DIR, 'best_regressor.pkl')
    if not os.path.exists(model_path):
        print("  Skipping regression scatter — best_regressor.pkl not found")
        return

    reg = joblib.load(model_path)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred = [], []
    for train_idx, test_idx in kf.split(X):
        reg.fit(X[train_idx], y[train_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(reg.predict(X[test_idx]))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    r2  = r2_score(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    r   = np.corrcoef(all_true, all_pred)[0, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(all_true, all_pred, alpha=0.6, color='#1976D2', edgecolors='white',
               linewidths=0.4, s=50)
    mn, mx = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.axvline(65, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(65, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xlabel('Actual Attention T-score', fontsize=12)
    ax.set_ylabel('Predicted Attention T-score', fontsize=12)
    ax.set_title('Regression: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.text(0.05, 0.92, f'R² = {r2:.3f}\nMAE = {mae:.2f}\nr = {r:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '4_regression_scatter.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 5. Feature importance bar chart ──────────────────────────────────────────

def plot_feature_importance(top_n: int = 20):
    imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    if not os.path.exists(imp_path):
        print("  Skipping feature importance — feature_importance.csv not found")
        return

    imp_df = pd.read_csv(imp_path).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
    bars = ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1],
                   color=colors, edgecolor='white')
    ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '5_feature_importance.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 6. SHAP summary plot ──────────────────────────────────────────────────────

def plot_shap_summary(X: np.ndarray, feature_cols: list):
    from sklearn.ensemble import RandomForestClassifier as RFC

    X_proc = preprocess_X(X)
    # Use subset for speed if large
    n = min(len(X_proc), 100)
    idx = np.random.RandomState(42).choice(len(X_proc), n, replace=False)
    X_sub = X_proc[idx]

    # Load best classifier or train fresh RF
    model_path = os.path.join(MODELS_DIR, 'best_classifier.pkl')
    if os.path.exists(model_path):
        clf_pipeline = joblib.load(model_path)
        # Extract the final estimator for SHAP
        clf = clf_pipeline.named_steps.get('clf', None)
        if clf is None:
            clf = RFC(n_estimators=100, random_state=42)
            clf.fit(X_proc, np.load(os.path.join(PROCESSED_DIR, 'y_class_tmp.npy'),
                                     allow_pickle=True))
    else:
        clf = RFC(n_estimators=100, random_state=42)
        df = pd.read_csv(os.path.join(PROCESSED_DIR, 'feature_matrix.csv'))
        y = df['attention_high'].values.astype(int)
        clf.fit(X_proc, y)

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sub)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        # Trim feature names to match preprocessed shape
        feat_names = feature_cols[:X_sub.shape[1]]

        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(shap_vals, X_sub, feature_names=feat_names,
                          show=False, max_display=20, plot_size=None)
        plt.title('SHAP Feature Importance — High Attention Class', fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, '6_shap_summary.png')
        plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  SHAP plot skipped: {e}")


# ── 7. Learning curve ─────────────────────────────────────────────────────────

def plot_learning_curve(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier as RFC

    X_proc = preprocess_X(X)
    clf = RFC(n_estimators=100, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X_proc, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='roc_auc', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, 'o-', color='#1976D2', label='Training AUC')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#1976D2')
    ax.plot(train_sizes, val_mean, 's-', color='#F44336', label='Validation AUC')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#F44336')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('AUC-ROC Score', fontsize=12)
    ax.set_title('Learning Curve (Random Forest)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim([0.4, 1.05])
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '7_learning_curve.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 8. Model comparison bar chart ─────────────────────────────────────────────

def plot_model_comparison():
    clf_path = os.path.join(MODELS_DIR, 'classification_results.csv')
    reg_path = os.path.join(MODELS_DIR, 'regression_results.csv')
    if not os.path.exists(clf_path) or not os.path.exists(reg_path):
        print("  Skipping model comparison — results CSVs not found")
        return

    clf_df = pd.read_csv(clf_path)
    reg_df = pd.read_csv(reg_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Classification comparison
    ax = axes[0]
    x = np.arange(len(clf_df))
    w = 0.25
    ax.bar(x - w, clf_df['Accuracy'], w, label='Accuracy', color='#42A5F5')
    ax.bar(x,     clf_df['F1'],       w, label='F1 Score',  color='#66BB6A')
    ax.bar(x + w, clf_df['AUC_ROC'],  w, label='AUC-ROC',   color='#EF5350')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_df['Model'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Model Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Regression comparison
    ax2 = axes[1]
    x2 = np.arange(len(reg_df))
    ax2.bar(x2 - 0.2, reg_df['R2'],       0.4, label='R²',          color='#AB47BC')
    ax2.bar(x2 + 0.2, reg_df['Pearson_r'], 0.4, label='Pearson r',   color='#FF7043')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(reg_df['Model'], rotation=15, ha='right', fontsize=9)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Regression Model Comparison', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '8_model_comparison.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 9. EEG feature correlation heatmap ───────────────────────────────────────

def plot_feature_correlation(df: pd.DataFrame):
    imp_path = os.path.join(MODELS_DIR, 'feature_importance.csv')
    if not os.path.exists(imp_path):
        return

    imp_df = pd.read_csv(imp_path).head(15)
    top_feats = imp_df['feature'].tolist()
    available = [f for f in top_feats if f in df.columns]
    if len(available) < 3:
        return

    sub_df = df[available + ['attention']].dropna()
    corr_matrix = sub_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.5,
                ax=ax, annot_kws={'size': 8})
    ax.set_title('Feature Correlation Heatmap (Top 15 Features)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '9_feature_correlation.png')
    plt.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── 10. Summary metrics table ─────────────────────────────────────────────────

def print_summary_table():
    clf_path = os.path.join(MODELS_DIR, 'classification_results.csv')
    reg_path = os.path.join(MODELS_DIR, 'regression_results.csv')
    if not os.path.exists(clf_path):
        return

    print("\n" + "="*65)
    print("CLASSIFICATION RESULTS (5-Fold CV)")
    print("="*65)
    clf_df = pd.read_csv(clf_path)
    print(clf_df[['Model', 'Accuracy', 'F1', 'AUC_ROC', 'Cohen_Kappa']].to_string(index=False))

    print("\n" + "="*65)
    print("REGRESSION RESULTS (5-Fold CV)")
    print("="*65)
    reg_df = pd.read_csv(reg_path)
    print(reg_df[['Model', 'MAE', 'R2', 'Pearson_r']].to_string(index=False))
    print("="*65)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_evaluation():
    print("\nGenerating evaluation plots...")
    df, X, y_class, y_reg, feature_cols = load_data()

    print("\n[1/9] Attention distribution...")
    plot_attention_distribution(df)

    print("[2/9] ROC curves...")
    plot_roc_curves(X, y_class)

    print("[3/9] Confusion matrix...")
    plot_confusion_matrix(X, y_class)

    print("[4/9] Regression scatter...")
    plot_regression_scatter(X, y_reg)

    print("[5/9] Feature importance bar chart...")
    plot_feature_importance()

    print("[6/9] SHAP summary...")
    plot_shap_summary(X, feature_cols)

    print("[7/9] Learning curve...")
    plot_learning_curve(X, y_class)

    print("[8/9] Model comparison...")
    plot_model_comparison()

    print("[9/9] Feature correlation heatmap...")
    plot_feature_correlation(df)

    print_summary_table()
    print(f"\nAll plots saved → {PLOTS_DIR}")


if __name__ == '__main__':
    run_evaluation()
