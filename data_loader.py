import pandas as pd
import numpy as np
import scipy.sparse as sp


def build_item_relative_loader(interaction_path, metadata_path, alpha=1.0):
    print("=" * 60)
    print("[*] LOADER V3: ITEM-RELATIVE CONFIDENCE SCALING")
    print("=" * 60)

    # 1. Đọc dữ liệu tương tác
    print("\n[1/4] Đang tải dữ liệu và tính toán Metadata...")
    cols = ['appid', 'author_steamid', 'voted_up', 'author_playtime_forever']
    df = pd.read_csv(interaction_path, usecols=cols)

    # 2. TÍNH TOÁN TRUNG VỊ (MEDIAN) CHO TỪNG GAME
    # Lưu ý: author.playtime_forever thường tính bằng phút
    print("[2/4] Đang tính toán Median playtime cho từng AppID...")

    # transform('median') giúp gán thẳng giá trị trung vị của game đó vào từng dòng tương tác
    df['median_playtime'] = df.groupby('appid')['author_playtime_forever'].transform('median')

    # Tránh chia cho 0: Nếu game chưa ai chơi hoặc median = 0, đặt mặc định là 1 phút
    df['median_playtime'] = df['median_playtime'].replace(0, 1)

    # 3. ÁP DỤNG CÔNG THỨC SỐ 2
    # c_ui = 1.0 + alpha * log(1 + t_ui / M_i)
    print("[3/4] Đang chuẩn hóa trọng số theo đặc thù từng Game...")
    df['weight'] = 1.0 + alpha * np.log1p(df['author_playtime_forever'] / df['median_playtime'])

    # Kiểm tra nhanh: Những game ngắn giờ sẽ có trọng số cao hơn nếu chơi quá mức trung bình
    print(f"  - Trọng số trung bình: {df['weight'].mean():.4f}")
    print(f"  - Trọng số max: {df['weight'].max():.4f}")

    # 4. KHỚP VỚI METADATA & TẠO CSR
    print("[4/4] Đang lọc Tags và tạo ma trận CSR...")
    df_meta = pd.read_csv(metadata_path)
    df_meta = df_meta.reset_index()
    df_meta = df_meta.rename(columns={'index': 'real_app_id'})
    df_meta['real_app_id'] = pd.to_numeric(df_meta['real_app_id'], errors='coerce')

    # Lọc bỏ các dòng lỗi và mất Tags
    df_meta = df_meta.dropna(subset=['real_app_id', 'Tags'])
    valid_apps = set(df_meta['real_app_id'].unique())


    # Lọc chỉ giữ lại tương tác của những game có Tags
    df_final = df[df['appid'].isin(valid_apps)].copy()

    # Mã hóa ID
    df_final['u_cat'] = df_final['author_steamid'].astype('category')
    df_final['i_cat'] = df_final['appid'].astype('category')

    user_map = dict(enumerate(df_final['u_cat'].cat.categories))
    item_map = dict(enumerate(df_final['i_cat'].cat.categories))

    u_ids = df_final['u_cat'].cat.codes.values
    i_ids = df_final['i_cat'].cat.codes.values
    weights = df_final['weight'].values.astype(np.float32)

    # Tách hành vi cho VEBPR (Bậc 1: Thích, Bậc 2: Không thích/View)
    is_positive = df_final['voted_up'].values == True

    play_csr = sp.csr_matrix((weights[is_positive],
                              (u_ids[is_positive], i_ids[is_positive])),
                             shape=(len(user_map), len(item_map)))
    view_csr = sp.csr_matrix((np.ones(sum(~is_positive), dtype=np.float32),
                              (u_ids[~is_positive], i_ids[~is_positive])),
                             shape=(len(user_map), len(item_map)))
    print(f"\n[OK] THANH CONG!")
    print(f"  - Đã xử lý {len(df_final):,} tương tác.")
    print(f"  - Kích thước CSR: {play_csr.shape}")
    print("=" * 60)

    return play_csr, view_csr, df_meta, user_map, item_map


def build_train_val_split(interaction_path, metadata_path, alpha=1.0):
    """Load data và split thành train/validation theo Leave-Last-Out.

    Strategy: Với mỗi user, giữ lại 1 interaction cuối cùng
    (theo timestamp_created) làm validation. Phần còn lại → train.

    Parameters
    ----------
    interaction_path : str
        Path tới file CSV tương tác.
    metadata_path : str
        Path tới file CSV metadata games.
    alpha : float
        Confidence scaling factor.

    Returns
    -------
    train_play_csr : csr_matrix
        Training Play matrix (positive feedback).
    train_view_csr : csr_matrix
        Training View matrix (weak positive feedback).
    val_dict : dict
        {user_internal_idx: [list of held-out item_internal_idx]}.
    combined_train_csr : csr_matrix
        Train play + view combined (cho BPR/ALS/UserKNN models).
    df_meta : DataFrame
        Game metadata.
    user_map : dict
        Internal user ID → original user ID.
    item_map : dict
        Internal item ID → original item ID.
    """
    print("=" * 60)
    print("[*] LOADER WITH TRAIN/VAL SPLIT (Leave-Last-Out)")
    print("=" * 60)

    # 1. Đọc dữ liệu tương tác (thêm timestamp_created)
    print("\n[1/5] Đang tải dữ liệu...")
    cols = ['appid', 'author_steamid', 'voted_up',
            'author_playtime_forever', 'timestamp_created']
    df = pd.read_csv(interaction_path, usecols=cols)

    # 2. Tính Median playtime & weight
    print("[2/5] Đang tính trọng số Item-Relative...")
    df['median_playtime'] = df.groupby('appid')['author_playtime_forever'].transform('median')
    df['median_playtime'] = df['median_playtime'].replace(0, 1)
    df['weight'] = 1.0 + alpha * np.log1p(df['author_playtime_forever'] / df['median_playtime'])

    # 3. Filter theo metadata (giữ games có Tags)
    print("[3/5] Đang lọc theo Metadata...")
    df_meta = pd.read_csv(metadata_path)
    df_meta = df_meta.reset_index()
    df_meta = df_meta.rename(columns={'index': 'real_app_id'})
    df_meta['real_app_id'] = pd.to_numeric(df_meta['real_app_id'], errors='coerce')
    df_meta = df_meta.dropna(subset=['real_app_id', 'Tags'])
    valid_apps = set(df_meta['real_app_id'].unique())
    df_final = df[df['appid'].isin(valid_apps)].copy()

    # Mã hóa ID
    df_final['u_cat'] = df_final['author_steamid'].astype('category')
    df_final['i_cat'] = df_final['appid'].astype('category')

    user_map = dict(enumerate(df_final['u_cat'].cat.categories))
    item_map = dict(enumerate(df_final['i_cat'].cat.categories))

    df_final['u_idx'] = df_final['u_cat'].cat.codes.values
    df_final['i_idx'] = df_final['i_cat'].cat.codes.values

    n_users = len(user_map)
    n_items = len(item_map)

    # 4. Leave-Last-Out Split
    print("[4/5] Đang thực hiện Leave-Last-Out Split...")
    df_final = df_final.sort_values('timestamp_created')

    # Lấy interaction cuối cùng của mỗi user làm validation
    val_indices = df_final.groupby('u_idx').tail(1).index
    is_val = df_final.index.isin(val_indices)

    df_train = df_final[~is_val].copy()
    df_val = df_final[is_val].copy()

    # Chỉ giữ val users có ít nhất 1 interaction trong train
    train_users = set(df_train['u_idx'].unique())
    df_val = df_val[df_val['u_idx'].isin(train_users)]

    # Build val_dict
    val_dict = {}
    for _, row in df_val.iterrows():
        u = int(row['u_idx'])
        i = int(row['i_idx'])
        if u not in val_dict:
            val_dict[u] = []
        val_dict[u].append(i)

    # 5. Build CSR matrices cho train
    print("[5/5] Đang xây dựng CSR matrices...")
    weights_train = df_train['weight'].values.astype(np.float32)
    u_train = df_train['u_idx'].values
    i_train = df_train['i_idx'].values
    is_positive_train = df_train['voted_up'].values == True

    # Play matrix (voted_up = True)
    train_play_csr = sp.csr_matrix(
        (weights_train[is_positive_train],
         (u_train[is_positive_train], i_train[is_positive_train])),
        shape=(n_users, n_items)
    )

    # View matrix (voted_up = False)
    n_neg = sum(~is_positive_train)
    train_view_csr = sp.csr_matrix(
        (np.ones(n_neg, dtype=np.float32),
         (u_train[~is_positive_train], i_train[~is_positive_train])),
        shape=(n_users, n_items)
    )

    # Combined matrix cho BPR/ALS/UserKNN (play + view, binary)
    combined_train_csr = sp.csr_matrix(
        (np.ones(len(u_train), dtype=np.float32),
         (u_train, i_train)),
        shape=(n_users, n_items)
    )

    print(f"\n[OK] SPLIT HOAN TAT!")
    print(f"  - Train: {len(df_train):,} interactions")
    print(f"  - Val: {len(df_val):,} users with held-out items")
    print(f"  - Play CSR: {train_play_csr.shape}, nnz={train_play_csr.nnz:,}")
    print(f"  - View CSR: {train_view_csr.shape}, nnz={train_view_csr.nnz:,}")
    print(f"  - Combined CSR: {combined_train_csr.shape}, nnz={combined_train_csr.nnz:,}")
    print("=" * 60)

    return (train_play_csr, train_view_csr, val_dict,
            combined_train_csr, df_meta, user_map, item_map)
