"""
Model Comparison Script.

Load best params from each Optuna study,
re-train each model with best params,
compare all metrics (NDCG@10, P@10, R@10, MAP@10, HR@10).
"""
import os
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import time

from data_loader import build_train_val_split
from evaluate import evaluate_model

# ===============================================================
INTERACTION_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\weighted_score_above_08.csv'
METADATA_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'
RESULTS_DIR = 'results'
EVAL_K = 10
NUM_EVAL_USERS = 10000
USE_GPU = False


def load_best_params(model_name):
    """Load best params from JSON file."""
    path = os.path.join(RESULTS_DIR, f'{model_name}_best.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get('best_params', {})


def build_model(model_name, params):
    """Create model instance from name and params."""
    if model_name == 'PopRank':
        from models.pop_model import PopModel
        return PopModel()

    elif model_name == 'ItemKNN':
        from models.userknn_model import ItemKNNModel
        return ItemKNNModel(K=params.get('K', 50))

    elif model_name == 'ALS':
        from models.als_model import ALSModel
        return ALSModel(
            factors=params.get('factors', 64),
            regularization=params.get('regularization', 0.01),
            alpha=params.get('alpha', 40.0),
            iterations=params.get('iterations', 15),
            use_gpu=USE_GPU,
            seed=42,
        )

    elif model_name == 'BPR':
        from models.bpr_model import BPRModel
        return BPRModel(
            factors=params.get('factors', 64),
            learning_rate=params.get('learning_rate', 0.01),
            regularization=params.get('regularization', 0.01),
            iterations=params.get('iterations', 100),
            use_gpu=USE_GPU,
            seed=42,
        )

    elif model_name == 'VEBPR':
        from models.vebpr_model import VEBPR
        return VEBPR(
            k=params.get('k', 64),
            max_iter=params.get('max_iter', 100),
            learning_rate=params.get('learning_rate', 0.01),
            lambda_reg=params.get('lambda_reg', 0.001),
            use_bias=True,
            seed=42,
            verbose=True,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == '__main__':
    print("=" * 70)
    print("[*] MODEL COMPARISON - Best Params from Optuna Tuning")
    print("=" * 70)

    # Load data
    (train_play_csr, train_view_csr, val_dict,
     combined_train_csr, df_meta, user_map, item_map) = build_train_val_split(
        INTERACTION_FILE, METADATA_FILE, alpha=1.0
    )

    model_names = ['PopRank', 'ItemKNN', 'ALS', 'BPR', 'VEBPR']
    all_results = []

    for name in model_names:
        print(f"\n{'-' * 70}")
        print(f"[>] {name}: Loading best params & re-training...")

        params = load_best_params(name)
        if params is None and name != 'PopRank':
            print(f"   [WARN] No tuning results found for {name}, skipping.")
            continue

        params = params or {}
        print(f"   Params: {params}")

        try:
            model = build_model(name, params)

            start = time.time()
            if name == 'VEBPR':
                model.fit(train_play_csr, view_csr=train_view_csr)
                train_for_eval = train_play_csr
            else:
                model.fit(combined_train_csr)
                train_for_eval = combined_train_csr

            train_time = time.time() - start

            # Full evaluation (more users)
            metrics = evaluate_model(model, train_for_eval, val_dict,
                                     K=EVAL_K, num_eval_users=NUM_EVAL_USERS)
            metrics['train_time'] = train_time
            metrics['model'] = name
            metrics['params'] = params
            all_results.append(metrics)

            print(f"   [OK] Done in {train_time:.1f}s")
            for key, val in metrics.items():
                if key not in ('model', 'params', 'train_time', 'num_evaluated_users'):
                    print(f"   {key}: {val:.6f}")

        except Exception as e:
            print(f"   [ERROR] {e}")
            continue

    # ===============================================================
    # COMPARISON TABLE
    # ===============================================================
    if all_results:
        all_results.sort(key=lambda x: x.get(f'NDCG@{EVAL_K}', 0), reverse=True)

        print(f"\n{'=' * 70}")
        print(f"[RESULTS] FINAL COMPARISON (K={EVAL_K})")
        print(f"{'=' * 70}")

        header = f"{'Rank':<5}{'Model':<10}{'NDCG@10':<10}{'P@10':<10}{'R@10':<10}{'MAP@10':<10}{'HR@10':<10}{'Time(s)':<8}"
        print(header)
        print("-" * 70)

        for rank, result in enumerate(all_results, 1):
            ndcg = result.get(f'NDCG@{EVAL_K}', 0)
            prec = result.get(f'Precision@{EVAL_K}', 0)
            rec = result.get(f'Recall@{EVAL_K}', 0)
            map_score = result.get(f'MAP@{EVAL_K}', 0)
            hr = result.get(f'HitRate@{EVAL_K}', 0)
            t = result.get('train_time', 0)

            print(f"  {rank:<3}{result['model']:<10}{ndcg:<10.6f}{prec:<10.6f}"
                  f"{rec:<10.6f}{map_score:<10.6f}{hr:<10.6f}{t:<8.1f}")

        # Save results
        output_path = os.path.join(RESULTS_DIR, 'comparison_full.json')
        with open(output_path, 'w') as f:
            clean_results = []
            for r in all_results:
                clean = {}
                for k, v in r.items():
                    if isinstance(v, (np.floating, np.integer)):
                        clean[k] = float(v)
                    else:
                        clean[k] = v
                clean_results.append(clean)
            json.dump(clean_results, f, indent=2)

        print(f"\n[SAVED] Results saved to: {output_path}")
        print(f"[BEST] Best model: {all_results[0]['model']} "
              f"(NDCG@{EVAL_K}={all_results[0].get(f'NDCG@{EVAL_K}', 0):.6f})")
