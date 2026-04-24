"""
ALS (Alternating Least Squares) / Weighted Matrix Factorization Model wrapper.
Sử dụng implicit library (GPU-accelerated).
"""
import numpy as np
from .base import BaseRecommender
from .userknn_model import _ensure_csr_compat


class ALSModel(BaseRecommender):
    """ALS/WMF — Matrix Factorization truyền thống.

    Dùng implicit.als.AlternatingLeastSquares làm engine.
    Confidence scaling: C = alpha * interaction_value.

    Parameters
    ----------
    factors : int, default=64
        Số chiều latent factors.
    regularization : float, default=0.01
        Regularization.
    alpha : float, default=40.0
        Confidence scaling factor.
    iterations : int, default=15
        Số ALS iterations.
    use_gpu : bool, default=True
        Sử dụng GPU (CUDA) nếu available.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, factors=64, regularization=0.01, alpha=40.0,
                 iterations=15, use_gpu=True, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.seed = seed
        self.model = None

    def get_name(self):
        return "ALS"

    def fit(self, train_csr, **kwargs):
        """Huấn luyện ALS model.

        Parameters
        ----------
        train_csr : csr_matrix (user × item)
            Interaction matrix. Values sẽ được scale bởi alpha.
        """
        from implicit.als import AlternatingLeastSquares

        train_csr = _ensure_csr_compat(train_csr)
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=self.use_gpu,
        )
        self.model.fit(train_csr, show_progress=False)
        return self

    def score(self, user_idx, item_idx=None):
        """Tính score cho user."""
        if item_idx is None:
            user_vec = self.model.user_factors[user_idx]
            scores = self.model.item_factors @ user_vec
            return np.array(scores).flatten()
        else:
            user_vec = self.model.user_factors[user_idx]
            item_vec = self.model.item_factors[item_idx]
            return float(np.dot(user_vec, item_vec))

    def recommend(self, user_idx, train_csr, N=10):
        """Top-N recommendation dùng implicit's built-in method."""
        train_csr = _ensure_csr_compat(train_csr)
        ids, scores = self.model.recommend(
            user_idx, train_csr[user_idx], N=N, filter_already_liked_items=True
        )
        return np.array(ids)
