"""
Popularity Baseline Model.
Non-personalized — recommend items theo số lượng interactions.
Dùng làm lower-bound baseline.
"""
import numpy as np
from .base import BaseRecommender


class PopModel(BaseRecommender):
    """Popularity Baseline — Non-personalized recommender.

    Recommend items có nhiều interactions nhất.
    Không có hyperparameters.
    """

    def __init__(self):
        self.item_popularity = None

    def get_name(self):
        return "PopRank"

    def fit(self, train_csr, **kwargs):
        """Tính popularity score cho mỗi item.

        Parameters
        ----------
        train_csr : csr_matrix (user × item)
        """
        # Popularity = số users đã tương tác với item
        self.item_popularity = np.array(train_csr.sum(axis=0)).flatten().astype(np.float32)
        return self

    def score(self, user_idx, item_idx=None):
        """Trả về popularity scores (giống nhau cho mọi user).

        Parameters
        ----------
        user_idx : int (bỏ qua — non-personalized)
        item_idx : int or None
        """
        if item_idx is None:
            return self.item_popularity.copy()
        else:
            return self.item_popularity[item_idx]
