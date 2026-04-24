import os
import sys
import json
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from data_loader import build_train_val_split
from models.vebpr_model import VEBPR

INTERACTION_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\weighted_score_above_08.csv'
METADATA_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'
BEST_PARAMS_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\results\VEBPR_best.json'
OUT_DIR = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\model_weights'

os.makedirs(OUT_DIR, exist_ok=True)

def export_vebpr():
    print("=" * 50)
    print(" BƯỚC 1: Đọc tham số tốt nhất từ kết quả Optuna")
    print("=" * 50)
    if not os.path.exists(BEST_PARAMS_FILE):
        print(f"[LỖI] Không tìm thấy file {BEST_PARAMS_FILE}")
        return

    with open(BEST_PARAMS_FILE, 'r') as f:
        best_res = json.load(f)
    
    params = best_res['best_params']
    print(f"Tham số: {params}")

    print("\n" + "=" * 50)
    print(" BƯỚC 2: Tải TOÀN BỘ dữ liệu (Full Dataset)")
    print("=" * 50)
    
    # Dùng combined_train_csr (toàn bộ Play) và train_view_csr (toàn bộ View)
    _, train_view_csr, _, combined_train_csr, _, _, item_map = build_train_val_split(
        INTERACTION_FILE, METADATA_FILE, alpha=1.0
    )

    print("\n" + "=" * 50)
    print(" BƯỚC 3: Train VEBPR trên toàn bộ dữ liệu")
    print("=" * 50)
    
    model = VEBPR(
        k=params['k'],
        max_iter=params['max_iter'],
        learning_rate=params['learning_rate'],
        lambda_reg=params['lambda_reg'],
        wt_ij=params.get('wt_ij', 1.0),
        wt_iv=params.get('wt_iv', 0.5),
        wt_vj=params.get('wt_vj', 0.5),
        use_bias=True,
        seed=42,
        verbose=True
    )

    # Train model (có thể mất khoảng 5-10 phút)
    model.fit(combined_train_csr, view_csr=train_view_csr)

    print("\n" + "=" * 50)
    print(" BƯỚC 4: Xuất trọng số (Export Weights)")
    print("=" * 50)
    
    # 1. Lưu ma trận I (Game Embeddings) và B (Biases)
    i_factors = model.i_factors
    i_biases = model.i_biases
    npz_path = os.path.join(OUT_DIR, 'vebpr_weights.npz')
    np.savez_compressed(npz_path, i_factors=i_factors, i_biases=i_biases)
    print(f"[OK] Đã lưu {npz_path}")

    # 2. Lưu Item Map (Đảo ngược để tra cứu từ Model Index -> Steam AppID)
    # item_map hiện tại là: {app_id: item_idx}
    # Ta cần lưu: {item_idx: app_id} và {app_id: item_idx} để tra cứu 2 chiều
    idx_to_appid = {int(idx): int(appid) for appid, idx in item_map.items()}
    appid_to_idx = {int(appid): int(idx) for appid, idx in item_map.items()}
    
    map_data = {
        'idx_to_appid': idx_to_appid,
        'appid_to_idx': appid_to_idx
    }
    
    map_path = os.path.join(OUT_DIR, 'item_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(map_data, f)
    print(f"[OK] Đã lưu {map_path}")

    print("\nHoàn tất Export Model! Chuyển sang bước inference.")

if __name__ == '__main__':
    export_vebpr()
