"""
Base interface cho tất cả recommender models.
Theo tư tưởng kiến trúc Cornac Recommender.
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):
    """Abstract base class for all recommender models.

    Mỗi model cần implement:
        - fit(): Huấn luyện model
        - score(): Tính điểm cho user-item pairs
        - recommend(): Trả về top-N items cho user
        - get_name(): Tên model
    """

    @abstractmethod
    def fit(self, train_csr, **kwargs):
        """Huấn luyện model trên interaction matrix.

        Parameters
        ----------
        train_csr : scipy.sparse.csr_matrix
            User-Item interaction matrix (CSR format).
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, user_idx, item_idx=None):
        """Tính điểm cho user-item pair(s).

        Parameters
        ----------
        user_idx : int
            Index của user.
        item_idx : int or None
            Index của item. Nếu None, trả về scores cho tất cả items.

        Returns
        -------
        float or np.ndarray
            Score(s) cho user-item pair(s).
        """
        raise NotImplementedError

    def recommend(self, user_idx, train_csr, N=10):
        """Trả về top-N recommended items cho user.

        Tự động loại bỏ các items user đã tương tác trong train.

        Parameters
        ----------
        user_idx : int
            Index của user.
        train_csr : scipy.sparse.csr_matrix
            Training interaction matrix (để loại bỏ seen items).
        N : int
            Số lượng items recommend.

        Returns
        -------
        np.ndarray
            Array các item indices, sắp xếp giảm dần theo score.
        """
        scores = self.score(user_idx)

        # Mask ra các items đã tương tác trong train set
        seen_items = train_csr[user_idx].indices
        scores[seen_items] = -np.inf

        # Top-N theo score giảm dần
        top_items = np.argpartition(scores, -N)[-N:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        return top_items

    @abstractmethod
    def get_name(self):
        """Trả về tên model.

        Returns
        -------
        str
        """
        raise NotImplementedError
