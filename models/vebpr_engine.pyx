# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
VEBPR Cython SGD Engine — Viết lại theo kiến trúc Cornac VEBPR.

Cải tiến:
    1. Lấy mẫu interaction-based (từ tập Play) thay vì random user.
    2. Negative sampling từ toàn bộ items (Uniform).
    3. View item sampling từ lịch sử View của chính user đó.
    4. Xử lý riêng trường hợp user không có view data -> Fallback về BPR.
    5. Kết hợp Time-weighted confidence (w_ui).
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.stdlib cimport rand, RAND_MAX, srand
from libcpp.algorithm cimport binary_search
from libcpp cimport bool

cdef inline int fast_rand() nogil:
    # Trên Windows, RAND_MAX chỉ là 32767 (15-bit). 
    # Cần ghép 2 số 15-bit để tạo ra số 30-bit.
    return (rand() << 15) ^ rand()

cdef bool _has_non_zero(int[:] indptr, int[:] indices,
                        int user_id, int item_id) noexcept nogil:
    cdef int start = indptr[user_id]
    cdef int end = indptr[user_id + 1]
    if end <= start:
        return False
    return binary_search(&indices[start], &indices[end], item_id)


cpdef tuple fit_vebpr_epoch(
        int[:] play_user_ids,
        int[:] play_indptr,
        int[:] play_indices,
        float[:] play_weights,
        int[:] view_indptr,
        int[:] view_indices,
        float[:, :] U,
        float[:, :] V,
        float[:] B,
        int num_users,
        int num_items,
        int num_factors,
        float lr,
        float reg,
        int num_samples,
        int num_play_interactions,
        float wt_ij = 1.0,
        float wt_iv = 1.0,
        float wt_vj = 1.0,
        int seed = 0,
):
    """Thực hiện 1 epoch SGD cho VEBPR theo sampling của Cornac."""
    if seed > 0:
        srand(seed)

    cdef int u, i, v, j, f, s
    cdef int num_view, pos_play_idx
    cdef float z_ij, z_iv, z_vj, sig_ij, sig_iv, sig_vj, w_ui
    cdef float u_f, i_f, v_f, j_f, grad_ij, grad_iv, grad_vj
    cdef long correct = 0, skipped = 0

    for s in range(num_samples):
        # 1. Sample Interaction (User u, Item i) dựa trên số lượt Play
        pos_play_idx = fast_rand() % num_play_interactions
        u = play_user_ids[pos_play_idx]
        i = play_indices[pos_play_idx]
        w_ui = play_weights[pos_play_idx]

        num_view = view_indptr[u + 1] - view_indptr[u]

        if num_view == 0:
            # Fallback to Normal BPR vì user này không có view data
            j = fast_rand() % num_items
            if _has_non_zero(play_indptr, play_indices, u, j):
                skipped += 1
                continue

            z_ij = B[i] - B[j]
            for f in range(num_factors):
                z_ij += U[u, f] * (V[i, f] - V[j, f])

            if z_ij > 0:
                correct += 1

            if z_ij > 15.0:
                sig_ij = 0.0
            elif z_ij < -15.0:
                sig_ij = 1.0
            else:
                sig_ij = 1.0 / (1.0 + exp(z_ij))

            # Trọng số BPR chuẩn (có kèm time weight)
            grad_ij = w_ui * sig_ij

            for f in range(num_factors):
                u_f = U[u, f]
                i_f = V[i, f]
                j_f = V[j, f]

                U[u, f] += lr * (grad_ij * (i_f - j_f) - reg * u_f)
                V[i, f] += lr * (grad_ij * u_f - reg * i_f)
                V[j, f] += lr * (-grad_ij * u_f - reg * j_f)

            B[i] += lr * (grad_ij - reg * B[i])
            B[j] += lr * (-grad_ij - reg * B[j])
            continue

        # 2. VEBPR: User có View Data
        v = view_indices[view_indptr[u] + (fast_rand() % num_view)]
        j = fast_rand() % num_items

        if _has_non_zero(play_indptr, play_indices, u, j) or _has_non_zero(view_indptr, view_indices, u, j):
            skipped += 1
            continue

        # Tính Scores
        z_ij = B[i] - B[j]
        z_iv = B[i] - B[v]
        z_vj = B[v] - B[j]
        for f in range(num_factors):
            z_ij += U[u, f] * (V[i, f] - V[j, f])
            z_iv += U[u, f] * (V[i, f] - V[v, f])
            z_vj += U[u, f] * (V[v, f] - V[j, f])

        if z_ij > 0 and z_iv > 0 and z_vj > 0:
            correct += 1

        # Sigmoids
        if z_ij > 15.0:
            sig_ij = 0.0
        elif z_ij < -15.0:
            sig_ij = 1.0
        else:
            sig_ij = 1.0 / (1.0 + exp(z_ij))

        if z_iv > 15.0:
            sig_iv = 0.0
        elif z_iv < -15.0:
            sig_iv = 1.0
        else:
            sig_iv = 1.0 / (1.0 + exp(z_iv))

        if z_vj > 15.0:
            sig_vj = 0.0
        elif z_vj < -15.0:
            sig_vj = 1.0
        else:
            sig_vj = 1.0 / (1.0 + exp(z_vj))

        # Áp dụng 3 hyper-parameters và time-weight (w_ui) cho Play
        grad_ij = wt_ij * w_ui * sig_ij
        grad_iv = wt_iv * w_ui * sig_iv
        grad_vj = wt_vj * 1.0 * sig_vj

        # Backward Pass
        for f in range(num_factors):
            u_f = U[u, f]
            i_f = V[i, f]
            v_f = V[v, f]
            j_f = V[j, f]

            U[u, f] += lr * (grad_ij * (i_f - j_f) + grad_iv * (i_f - v_f) + grad_vj * (v_f - j_f) - reg * u_f)
            V[i, f] += lr * (grad_ij * u_f + grad_iv * u_f - reg * i_f)
            V[v, f] += lr * (-grad_iv * u_f + grad_vj * u_f - reg * v_f)
            V[j, f] += lr * (-grad_ij * u_f - grad_vj * u_f - reg * j_f)

        B[i] += lr * (grad_ij + grad_iv - reg * B[i])
        B[v] += lr * (-grad_iv + grad_vj - reg * B[v])
        B[j] += lr * (-grad_ij - grad_vj - reg * B[j])

    return correct, skipped
