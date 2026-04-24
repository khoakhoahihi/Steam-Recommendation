"""
Item-KNN Collaborative Filtering Model wrapper.
Uses sklearn cosine_similarity on item-item (19K items, much faster than user-user 334K).
"""
import numpy as np
import scipy.sparse as sp
from .base import BaseRecommender


def _ensure_csr_compat(csr_matrix):
    """Ensure CSR matrix compatibility with implicit library.

    implicit on Windows requires int32 indices, not int64 (long long).
    Also ensures data is float32.
    """
    needs_rebuild = False
    indices = csr_matrix.indices
    indptr = csr_matrix.indptr
    data = csr_matrix.data

    if indices.dtype != np.int32:
        indices = indices.astype(np.int32)
        needs_rebuild = True
    if indptr.dtype != np.int32:
        indptr = indptr.astype(np.int32)
        needs_rebuild = True
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        needs_rebuild = True

    if needs_rebuild:
        csr_matrix = sp.csr_matrix(
            (data, indices, indptr),
            shape=csr_matrix.shape
        )
    return csr_matrix


class ItemKNNModel(BaseRecommender):
    """Item-KNN CF - Memory-based Collaborative Filtering.

    Computes item-item cosine similarity and recommends items
    most similar to user's previously interacted items.

    Much faster than User-KNN for datasets with many users and fewer items.
    (19K items vs 334K users)

    Parameters
    ----------
    K : int, default=50
        Number of nearest item neighbors to consider.
    """

    def __init__(self, K=50):
        self.K = K
        self._train_csr = None
        self._item_sim = None  # item-item similarity (n_items x n_items sparse)

    def get_name(self):
        return "ItemKNN"

    def fit(self, train_csr, **kwargs):
        """Train ItemKNN model by computing item-item cosine similarity.

        Parameters
        ----------
        train_csr : csr_matrix (n_users x n_items)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        self._train_csr = train_csr.tocsr()
        n_items = train_csr.shape[1]

        # Item-item similarity: transpose to get items as rows
        item_features = train_csr.T.tocsr()  # (n_items x n_users)

        # Compute full item-item cosine similarity
        # For 19K items this is manageable (~1.4GB dense, but we use sparse)
        print(f"   Computing item-item similarity for {n_items} items...")

        # Compute in batches for memory efficiency
        batch_size = 2000
        rows, cols, vals = [], [], []

        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            batch = item_features[start:end]

            # Cosine similarity of batch items vs all items
            sim_batch = cosine_similarity(batch, item_features, dense_output=True)

            for local_i in range(sim_batch.shape[0]):
                global_i = start + local_i
                row = sim_batch[local_i]
                row[global_i] = 0  # exclude self-similarity

                # Keep only top-K neighbors
                if self.K < len(row):
                    top_k_idx = np.argpartition(row, -self.K)[-self.K:]
                    for j in top_k_idx:
                        if row[j] > 0:
                            rows.append(global_i)
                            cols.append(j)
                            vals.append(row[j])

        self._item_sim = sp.csr_matrix(
            (np.array(vals, dtype=np.float32),
             (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            shape=(n_items, n_items)
        )
        print(f"   Item similarity matrix: {self._item_sim.shape}, nnz={self._item_sim.nnz:,}")

        return self

    def score(self, user_idx, item_idx=None):
        """Compute scores based on item-item similarity.

        For each candidate item j, score = sum of sim(j, i) for all items i
        that the user has interacted with.
        """
        if item_idx is None:
            # Get user's interacted items
            user_items = self._train_csr[user_idx]  # sparse row

            # Score = user_items @ item_sim.T
            # For each item j: score_j = sum(user_items[i] * sim[i, j])
            scores = user_items.dot(self._item_sim).toarray().flatten().astype(np.float32)
            return scores
        else:
            user_items = self._train_csr[user_idx]
            return float(user_items.dot(self._item_sim[:, item_idx]).toarray().flatten()[0])
