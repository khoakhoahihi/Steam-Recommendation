"""
Evaluation Module cho Ranking Recommender Systems.

Metrics:
    - NDCG@K (Normalized Discounted Cumulative Gain) — Primary metric
    - Precision@K
    - Recall@K
    - MAP@K (Mean Average Precision)
    - HitRate@K

Strategy: Leave-Last-Out per user
"""
import numpy as np
from collections import defaultdict


def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain @ K.

    Parameters
    ----------
    recommended : list/array
        Danh sách item IDs được recommend (đã sắp xếp theo score giảm dần).
    relevant : set
        Tập các item IDs thực sự relevant (ground truth).
    k : int
        Cutoff position.

    Returns
    -------
    float : NDCG score [0, 1]
    """
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 vì index từ 0, log2(1) = 0

    # Ideal DCG: tất cả relevant items ở top
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended, relevant, k):
    """Precision @ K.

    Parameters
    ----------
    recommended : list/array
    relevant : set
    k : int

    Returns
    -------
    float : Precision score [0, 1]
    """
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    """Recall @ K.

    Parameters
    ----------
    recommended : list/array
    relevant : set
    k : int

    Returns
    -------
    float : Recall score [0, 1]
    """
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def map_at_k(recommended, relevant, k):
    """Mean Average Precision @ K (per user).

    Parameters
    ----------
    recommended : list/array
    relevant : set
    k : int

    Returns
    -------
    float : AP score [0, 1]
    """
    rec_k = recommended[:k]
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant), k) if len(relevant) > 0 else 0.0


def hit_rate_at_k(recommended, relevant, k):
    """Hit Rate @ K (binary: có ít nhất 1 hit hay không).

    Parameters
    ----------
    recommended : list/array
    relevant : set
    k : int

    Returns
    -------
    float : 0.0 hoặc 1.0
    """
    rec_k = recommended[:k]
    return 1.0 if any(item in relevant for item in rec_k) else 0.0


def evaluate_model(model, train_csr, val_dict, K=10, num_eval_users=None):
    """Đánh giá model trên validation set.

    Parameters
    ----------
    model : BaseRecommender
        Model đã được fit.
    train_csr : csr_matrix
        Training interaction matrix (dùng để filter seen items).
    val_dict : dict
        {user_idx: [list of held-out item_ids]} — validation set.
    K : int, default=10
        Cutoff position cho tất cả metrics.
    num_eval_users : int or None
        Số users tối đa để evaluate (sampling để tăng tốc).
        None = evaluate tất cả.

    Returns
    -------
    dict : {metric_name: float} — Mean metrics across users.
    """
    metrics = defaultdict(list)

    eval_users = list(val_dict.keys())
    if num_eval_users is not None and num_eval_users < len(eval_users):
        rng = np.random.RandomState(42)
        eval_users = rng.choice(eval_users, size=num_eval_users, replace=False).tolist()

    for user_idx in eval_users:
        relevant = set(val_dict[user_idx])
        if len(relevant) == 0:
            continue

        # Lấy top-K recommendations
        try:
            recommended = model.recommend(user_idx, train_csr, N=K)
        except Exception:
            continue

        # Tính tất cả metrics
        metrics['NDCG@{}'.format(K)].append(ndcg_at_k(recommended, relevant, K))
        metrics['Precision@{}'.format(K)].append(precision_at_k(recommended, relevant, K))
        metrics['Recall@{}'.format(K)].append(recall_at_k(recommended, relevant, K))
        metrics['MAP@{}'.format(K)].append(map_at_k(recommended, relevant, K))
        metrics['HitRate@{}'.format(K)].append(hit_rate_at_k(recommended, relevant, K))

    # Mean across users
    result = {}
    for key, values in metrics.items():
        result[key] = float(np.mean(values)) if len(values) > 0 else 0.0

    result['num_evaluated_users'] = len(metrics.get('NDCG@{}'.format(K), []))
    return result
