"""
Optuna Multi-Model Hyperparameter Tuning Engine.

Tối ưu NDCG@10 cho 5 models:
    1. VEBPR (View-Enhanced BPR) — Cython SGD
    2. BPR (Bayesian Personalized Ranking) — implicit
    3. ALS (Alternating Least Squares) — implicit
    4. UserKNN (Cosine CF) — implicit
    5. PopRank (Popularity Baseline) — no tuning

Config: 30 trials/model, TPE Sampler, MedianPruner
Hardware: GTX 3050 + 16GB RAM
"""
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Fix OpenBLAS threading warning

# Fix Windows cp1252 encoding for Vietnamese text in data_loader prints
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import optuna
import numpy as np
import time
import json

from data_loader import build_train_val_split
from evaluate import evaluate_model

# ===============================================================
# CAU HINH CHUNG
# ===============================================================
INTERACTION_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\weighted_score_above_08.csv'
METADATA_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'
RESULTS_DIR = 'results'
N_TRIALS = 30          # Trials per model
EVAL_K = 10            # NDCG@10
NUM_EVAL_USERS = 5000  # Sample users cho evaluation (tang toc)
USE_GPU = False        # implicit CUDA not available, use CPU

os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# LOAD DATA MỘT LẦN DUY NHẤT
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("[*] KHOI DONG HE THONG TU DONG TIM SIEU THAM SO (OPTUNA)")
print("=" * 60)

(train_play_csr, train_view_csr, val_dict,
 combined_train_csr, df_meta, user_map, item_map) = build_train_val_split(
    INTERACTION_FILE, METADATA_FILE, alpha=1.0
)

print(f"\n[INFO] Validation set: {len(val_dict):,} users")
print(f"[INFO] Eval sample: {NUM_EVAL_USERS:,} users")
print(f"[INFO] Metric: NDCG@{EVAL_K}")
print(f"[INFO] Trials per model: {N_TRIALS}")


# ═══════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def objective_vebpr(trial):
    """Objective cho VEBPR model."""
    from models.vebpr_model import VEBPR

    k = trial.suggest_categorical('k', [32, 64, 128])
    lr = trial.suggest_float('learning_rate', 1e-3, 0.05, log=True)
    reg = trial.suggest_float('lambda_reg', 1e-4, 0.01, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 500, step=50)
    wt_ij = trial.suggest_float('wt_ij', 0.1, 1.0)
    wt_iv = trial.suggest_float('wt_iv', 0.0, 1.0)
    wt_vj = trial.suggest_float('wt_vj', 0.0, 1.0)

    model = VEBPR(
        k=k,
        max_iter=max_iter,
        learning_rate=lr,
        lambda_reg=reg,
        wt_ij=wt_ij,
        wt_iv=wt_iv,
        wt_vj=wt_vj,
        use_bias=True,
        seed=42,
        verbose=False,
    )
    model.fit(train_play_csr, view_csr=train_view_csr)

    metrics = evaluate_model(model, train_play_csr, val_dict,
                             K=EVAL_K, num_eval_users=NUM_EVAL_USERS)
    return metrics[f'NDCG@{EVAL_K}']


def objective_bpr(trial):
    """Objective cho BPR model (implicit)."""
    from models.bpr_model import BPRModel

    factors = trial.suggest_int('factors', 32, 256, step=32)
    lr = trial.suggest_float('learning_rate', 1e-4, 0.1, log=True)
    reg = trial.suggest_float('regularization', 1e-5, 0.1, log=True)
    iterations = trial.suggest_int('iterations', 50, 300, step=25)

    model = BPRModel(
        factors=factors,
        learning_rate=lr,
        regularization=reg,
        iterations=iterations,
        use_gpu=USE_GPU,
        seed=42,
    )
    model.fit(combined_train_csr)

    metrics = evaluate_model(model, combined_train_csr, val_dict,
                             K=EVAL_K, num_eval_users=NUM_EVAL_USERS)
    return metrics[f'NDCG@{EVAL_K}']


def objective_als(trial):
    """Objective cho ALS/WMF model (implicit)."""
    from models.als_model import ALSModel

    factors = trial.suggest_int('factors', 32, 256, step=32)
    reg = trial.suggest_float('regularization', 1e-3, 10.0, log=True)
    alpha = trial.suggest_float('alpha', 1.0, 100.0, log=True)
    iterations = trial.suggest_int('iterations', 15, 50, step=5)

    model = ALSModel(
        factors=factors,
        regularization=reg,
        alpha=alpha,
        iterations=iterations,
        use_gpu=USE_GPU,
        seed=42,
    )
    model.fit(combined_train_csr)

    metrics = evaluate_model(model, combined_train_csr, val_dict,
                             K=EVAL_K, num_eval_users=NUM_EVAL_USERS)
    return metrics[f'NDCG@{EVAL_K}']


def objective_userknn(trial):
    """Objective cho ItemKNN CF model."""
    from models.userknn_model import ItemKNNModel

    K = trial.suggest_int('K', 10, 200, step=10)

    model = ItemKNNModel(K=K)
    model.fit(combined_train_csr)

    metrics = evaluate_model(model, combined_train_csr, val_dict,
                             K=EVAL_K, num_eval_users=NUM_EVAL_USERS)
    return metrics[f'NDCG@{EVAL_K}']



# ═══════════════════════════════════════════════════════════
# TUNING PIPELINE
# ═══════════════════════════════════════════════════════════

def run_study(name, objective, n_trials):
    """Chạy Optuna study cho 1 model.

    Parameters
    ----------
    name : str
        Tên model.
    objective : callable
        Optuna objective function.
    n_trials : int
        Số trials.

    Returns
    -------
    study : optuna.Study
    """
    print(f"\n{'=' * 60}")
    print(f"[TUNING] {name} ({n_trials} trials)")
    print(f"{'=' * 60}")

    study = optuna.create_study(
        study_name=name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"\n[OK] {name} TUNING HOAN TAT ({elapsed:.1f}s)")
    print(f"   Best NDCG@{EVAL_K}: {study.best_value:.6f}")
    print(f"   Best Params: {study.best_params}")

    # Lưu kết quả
    result = {
        'model': name,
        'best_ndcg': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'elapsed_seconds': elapsed,
    }
    with open(os.path.join(RESULTS_DIR, f'{name}_best.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return study


def evaluate_poprank():
    """Danh gia PopRank baseline (khong can tuning)."""
    from models.pop_model import PopModel

    print(f"\n{'=' * 60}")
    print(f"[EVAL] PopRank Baseline (no tuning)")
    print(f"{'=' * 60}")

    model = PopModel()
    model.fit(combined_train_csr)

    metrics = evaluate_model(model, combined_train_csr, val_dict,
                             K=EVAL_K, num_eval_users=NUM_EVAL_USERS)

    print(f"   NDCG@{EVAL_K}: {metrics[f'NDCG@{EVAL_K}']:.6f}")
    print(f"   Precision@{EVAL_K}: {metrics[f'Precision@{EVAL_K}']:.6f}")
    print(f"   HitRate@{EVAL_K}: {metrics[f'HitRate@{EVAL_K}']:.6f}")

    result = {
        'model': 'PopRank',
        'best_ndcg': metrics[f'NDCG@{EVAL_K}'],
        'best_params': {},
        'all_metrics': metrics,
    }
    with open(os.path.join(RESULTS_DIR, 'PopRank_best.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return metrics


# ═══════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_studies = {}

    # 1. PopRank Baseline (no tuning)
    pop_metrics = evaluate_poprank()

    # 2. ItemKNN CF
    try:
        all_studies['ItemKNN'] = run_study('ItemKNN', objective_userknn, N_TRIALS)
    except Exception as e:
        print(f"\n[ERROR] ItemKNN failed: {e}")

    # 3. ALS / WMF
    try:
        all_studies['ALS'] = run_study('ALS', objective_als, N_TRIALS)
    except Exception as e:
        print(f"\n[ERROR] ALS failed: {e}")

    # 4. BPR
    try:
        all_studies['BPR'] = run_study('BPR', objective_bpr, N_TRIALS)
    except Exception as e:
        print(f"\n[ERROR] BPR failed: {e}")

    # 5. VEBPR (slowest -> last)
    try:
        all_studies['VEBPR'] = run_study('VEBPR', objective_vebpr, N_TRIALS)
    except Exception as e:
        print(f"\n[ERROR] VEBPR failed: {e}")


    # ═══════════════════════════════════════════════════════
    # TỔNG KẾT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] TONG KET TUNING - NDCG@{EVAL_K}")
    print(f"{'=' * 60}")

    summary = []
    summary.append(('PopRank', pop_metrics[f'NDCG@{EVAL_K}'], {}))

    for name, study in all_studies.items():
        summary.append((name, study.best_value, study.best_params))

    # Sort theo NDCG giảm dần
    summary.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6}{'Model':<12}{'NDCG@10':<12}{'Best Params'}")
    print("-" * 60)
    for rank, (name, ndcg, params) in enumerate(summary, 1):
        params_str = ', '.join(f'{k}={v}' for k, v in params.items()) if params else '-'
        print(f"  {rank:<4}{name:<12}{ndcg:<12.6f}{params_str}")

    # Lưu summary
    summary_data = [
        {'rank': r + 1, 'model': n, 'ndcg_10': v, 'best_params': p}
        for r, (n, v, p) in enumerate(summary)
    ]
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[SAVED] Ket qua da luu vao thu muc '{RESULTS_DIR}/'")
    print(f"[BEST] Model tot nhat: {summary[0][0]} (NDCG@{EVAL_K}={summary[0][1]:.6f})")
