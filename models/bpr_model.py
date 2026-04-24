"""
BPR (Bayesian Personalized Ranking) Model wrapper.
Sử dụng implicit library (GPU-accelerated).
"""
import numpy as np
from .base import BaseRecommender
from .userknn_model import _ensure_csr_compat


class BPRModel(BaseRecommender):
    """BPR truyền thống 2-level (Positive > Unobserved).

    Dùng implicit.bpr.BayesianPersonalizedRanking làm engine.

    Parameters
    ----------
    factors : int, default=64
        Số chiều latent factors.
    learning_rate : float, default=0.01
        Learning rate.
    regularization : float, default=0.01
        Regularization.
    iterations : int, default=100
        Số epochs.
    use_gpu : bool, default=True
        Sử dụng GPU (CUDA) nếu available.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, factors=64, learning_rate=0.01, regularization=0.01,
                 iterations=100, use_gpu=True, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.seed = seed
        self.model = None

    def get_name(self):
        return "BPR"

    def fit(self, train_csr, **kwargs):
        """Huấn luyện BPR model.

        Parameters
        ----------
        train_csr : csr_matrix (user × item)
            Interaction matrix.
        """
        from implicit.bpr import BayesianPersonalizedRanking

        train_csr = _ensure_csr_compat(train_csr)
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.seed,
            use_gpu=self.use_gpu,
        )
        # implicit expects item-user matrix (CSC), but fit() handles transpose
        self.model.fit(train_csr, show_progress=False)
        return self

    def score(self, user_idx, item_idx=None):
        """Tính score cho user."""
        if item_idx is None:
            user_vec = self.model.user_factors[user_idx]
            scores = self.model.item_factors @ user_vec
            # Thêm item biases nếu có
            if hasattr(self.model, 'item_biases') and self.model.item_biases is not None:
                scores += self.model.item_biases
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
