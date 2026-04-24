"""
VEBPR (View-Enhanced BPR) Model — Python Wrapper.

Kiến trúc theo tư tưởng Cornac BPR:
    __init__() → _init() → _prepare_data() → fit() → score() → recommend()

Đặc trưng so với BPR chuẩn:
    - 3-level feedback: Play > View > Unobserved
    - Time-weighted confidence: c_ui = 1 + α·log(1 + t_ui / M_i)
    - Item bias vector
"""
import numpy as np
import time

from .base import BaseRecommender

# Cython engine sẽ được import sau khi compile
try:
    from models import vebpr_engine
except ImportError:
    vebpr_engine = None


class VEBPR(BaseRecommender):
    """View-Enhanced Bayesian Personalized Ranking.

    Parameters
    ----------
    k : int, default=64
        Số chiều latent factors.

    max_iter : int, default=100
        Số epochs tối đa.

    learning_rate : float, default=0.01
        Learning rate cho SGD.

    lambda_reg : float, default=0.001
        Hệ số regularization.

    use_bias : bool, default=True
        Sử dụng item bias hay không.

    seed : int or None, default=None
        Random seed cho reproducibility.

    verbose : bool, default=True
        Hiển thị training logs.
    """

    def __init__(
        self,
        k=64,
        max_iter=100,
        learning_rate=0.01,
        lambda_reg=0.001,
        wt_ij=1.0,
        wt_iv=1.0,
        wt_vj=1.0,
        use_bias=True,
        seed=None,
        verbose=True,
    ):
        self.k = int(k)
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.wt_ij = wt_ij
        self.wt_iv = wt_iv
        self.wt_vj = wt_vj
        self.use_bias = use_bias
        self.seed = seed
        self.verbose = verbose

        # Sẽ được khởi tạo trong _init()
        self.u_factors = None
        self.i_factors = None
        self.i_biases = None
        self.num_users = 0
        self.num_items = 0

    def get_name(self):
        return "VEBPR"

    def _init(self, n_users, n_items):
        """Khởi tạo latent factors theo Cornac-style.

        Dùng uniform initialization: U(−0.5/k, 0.5/k)
        """
        self.num_users = n_users
        self.num_items = n_items

        rng = np.random.RandomState(self.seed)
        scale = 0.5 / self.k

        self.u_factors = rng.uniform(-scale, scale, size=(n_users, self.k)).astype(np.float32)
        self.i_factors = rng.uniform(-scale, scale, size=(n_items, self.k)).astype(np.float32)
        self.i_biases = np.zeros(n_items, dtype=np.float32)

    def _prepare_data(self, play_csr, view_csr):
        """Chuẩn bị C-arrays từ CSR matrices cho Cython engine.

        Parameters
        ----------
        play_csr : csr_matrix
            Play (positive) interaction matrix, data = confidence weights.
        view_csr : csr_matrix
            View (weak positive) interaction matrix.

        Returns
        -------
        dict : Các arrays đã ép kiểu int32/float32 cho Cython.
        """
        # Đảm bảo sorted indices cho binary_search
        play_csr.sort_indices()
        view_csr.sort_indices()

        # Tạo play_user_ids array cho việc lấy mẫu theo Interaction (Cornac-style)
        play_counts = np.ediff1d(play_csr.indptr)
        play_user_ids = np.repeat(np.arange(play_csr.shape[0]), play_counts).astype(np.int32)

        return {
            'play_user_ids': play_user_ids,
            'play_indptr': play_csr.indptr.astype(np.int32),
            'play_indices': play_csr.indices.astype(np.int32),
            'play_weights': play_csr.data.astype(np.float32),
            'view_indptr': view_csr.indptr.astype(np.int32),
            'view_indices': view_csr.indices.astype(np.int32),
        }

    def fit(self, train_csr, view_csr=None, **kwargs):
        """Huấn luyện VEBPR model.

        Parameters
        ----------
        train_csr : csr_matrix
            Play interaction matrix (positive feedback).
        view_csr : csr_matrix or None
            View interaction matrix (weak positive feedback).
            Nếu None, model hoạt động như BPR thường.

        Returns
        -------
        self
        """
        import scipy.sparse as sp

        if vebpr_engine is None:
            raise ImportError(
                "Cython engine chưa được compile. "
                "Chạy: python setup.py build_ext --inplace"
            )

        n_users, n_items = train_csr.shape

        # Nếu không có view data, tạo matrix rỗng
        if view_csr is None:
            view_csr = sp.csr_matrix((n_users, n_items), dtype=np.float32)

        self._init(n_users, n_items)
        data = self._prepare_data(train_csr, view_csr)

        # Seed is handled internally by numpy rng for initialization
        # The Cython engine uses C rand() — seed via np.random
        np.random.seed(self.seed if self.seed is not None else 42)

        num_samples = train_csr.nnz  # samples per epoch (giống Cornac)

        if self.verbose:
            print(f"[VEBPR] Training: K={self.k}, lr={self.learning_rate}, "
                  f"reg={self.lambda_reg}, epochs={self.max_iter}")
            print(f"        Weights: ij={self.wt_ij}, iv={self.wt_iv}, vj={self.wt_vj}")
            print(f"   Users={n_users:,}, Items={n_items:,}, "
                  f"Play={train_csr.nnz:,}, View={view_csr.nnz:,}")

        start_time = time.time()

        for epoch in range(self.max_iter):
            correct, skipped = vebpr_engine.fit_vebpr_epoch(
                data['play_user_ids'],
                data['play_indptr'],
                data['play_indices'],
                data['play_weights'],
                data['view_indptr'],
                data['view_indices'],
                self.u_factors,
                self.i_factors,
                self.i_biases,
                n_users, n_items, self.k,
                self.learning_rate,
                self.lambda_reg,
                num_samples,
                train_csr.nnz,  # num_play_interactions
                self.wt_ij,
                self.wt_iv,
                self.wt_vj,
            )

            if self.verbose and (epoch + 1) % max(1, self.max_iter // 10) == 0:
                total = num_samples - skipped
                acc = 100.0 * correct / max(total, 1)
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch + 1:>4d}/{self.max_iter} | "
                      f"correct={acc:.1f}% | skipped={skipped} | "
                      f"time={elapsed:.1f}s")

        if self.verbose:
            total_time = time.time() - start_time
            print(f"[OK] VEBPR training done in {total_time:.1f}s")

        return self

    def score(self, user_idx, item_idx=None):
        """Tính score cho user-item pair(s).

        Cornac-style: score = bias_i + U[u] · V[i]

        Parameters
        ----------
        user_idx : int
        item_idx : int or None

        Returns
        -------
        float or np.ndarray
        """
        if item_idx is None:
            # Score cho tất cả items
            scores = np.copy(self.i_biases) if self.use_bias else np.zeros(self.num_items, dtype=np.float32)
            scores += self.u_factors[user_idx] @ self.i_factors.T
            return scores
        else:
            score = self.i_biases[item_idx] if self.use_bias else 0.0
            score += np.dot(self.u_factors[user_idx], self.i_factors[item_idx])
            return score
