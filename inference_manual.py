import os
import json
import numpy as np
import pandas as pd
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MODEL_DIR = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\model_weights'
METADATA_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'

def load_model():
    print("[1] Đang tải mô hình VEBPR...")
    npz_path = os.path.join(MODEL_DIR, 'vebpr_weights.npz')
    map_path = os.path.join(MODEL_DIR, 'item_map.json')
    
    if not os.path.exists(npz_path) or not os.path.exists(map_path):
        raise FileNotFoundError("Chưa tìm thấy model_weights. Vui lòng chạy export_model.py trước.")
        
    data = np.load(npz_path)
    i_factors = data['i_factors']
    i_biases = data['i_biases']
    
    with open(map_path, 'r', encoding='utf-8') as f:
        item_map = json.load(f)
        
    # item_map['idx_to_appid'] chứa {appid: idx}
    # item_map['appid_to_idx'] chứa {idx: appid}
    return i_factors, i_biases, item_map['idx_to_appid'], item_map['appid_to_idx']

def load_game_database():
    print("[2] Đang tải cơ sở dữ liệu Game...")
    df = pd.read_csv(METADATA_FILE)
    df.reset_index(inplace=True)
    # df['index'] là AppID thật, df['AppID'] là Name (do lỗi lệch cột của CSV)
    
    # Tạo map appid -> title và title (lowercase) -> appid
    appid_to_title = {int(row['index']): str(row['AppID']) for _, row in df.iterrows()}
    title_to_appid = {str(row['AppID']).lower().strip(): int(row['index']) for _, row in df.iterrows()}
    
    return appid_to_title, title_to_appid

def fold_in_user(selected_appids, i_factors, i_biases, appid_to_idx, epochs=50, lr=0.01, reg=0.01):
    """
    Folding-in: Tính toán u_factor dựa trên các game người dùng đã chọn.
    """
    k = i_factors.shape[1]
    num_items = i_factors.shape[0]
    
    played_indices = []
    # Vì là chọn tay nên ta mặc định mỗi game có trọng số cao (ví dụ: 3.0 tương đương chơi nhiều)
    for app_id in selected_appids:
        app_id_str = str(app_id)
        if app_id_str in appid_to_idx:
            played_indices.append(appid_to_idx[app_id_str])
            
    if not played_indices:
        return None, []
        
    played_indices = np.array(played_indices, dtype=np.int32)
    
    # Khởi tạo
    np.random.seed(42)
    scale = 0.5 / k
    u_factor = np.random.uniform(-scale, scale, size=k).astype(np.float32)
    
    # SGD (Học nhanh sở thích từ danh sách chọn tay)
    for epoch in range(epochs):
        for i in played_indices:
            j = np.random.randint(0, num_items)
            while j in played_indices:
                j = np.random.randint(0, num_items)
            
            z = u_factor.dot(i_factors[i] - i_factors[j]) + i_biases[i] - i_biases[j]
            z_clipped = np.clip(z, -15.0, 15.0)
            sigmoid = 1.0 / (1.0 + np.exp(z_clipped))
            
            # Trọng số mặc định là 3.0 cho các game tự chọn (thể hiện sự yêu thích rõ ràng)
            grad = 3.0 * sigmoid
            u_factor += lr * (grad * (i_factors[i] - i_factors[j]) - reg * u_factor)
            
    return u_factor, played_indices

def recommend(u_factor, i_factors, i_biases, played_indices, idx_to_appid, appid_to_title, N=10):
    scores = i_factors.dot(u_factor) + i_biases
    scores[played_indices] = -np.inf
    top_indices = np.argsort(scores)[::-1][:N]
    
    print("\n" + "="*50)
    print(" DANH SÁCH GAME GỢI Ý CHO BẠN")
    print("="*50)
    for rank, idx in enumerate(top_indices, 1):
        app_id = int(idx_to_appid[str(idx)])
        title = appid_to_title.get(app_id, f"AppID: {app_id}")
        print(f"#{rank:<2} | {title}")

def main():
    # 1. Load data
    try:
        i_factors, i_biases, map_appid_to_idx, map_idx_to_appid = load_model()
        appid_to_title, title_to_appid = load_game_database()
    except Exception as e:
        print(f"Lỗi: {e}")
        return

    print("\n--- HỆ THỐNG GỢI Ý GAME THÔNG MINH (MANUAL MODE) ---")
    print("Nhập tên các game bạn thích (nhập 'done' để kết thúc):")
    
    selected_appids = []
    while True:
        query = input("> ").strip().lower()
        if query == 'done':
            break
        if not query:
            continue
            
        # Tìm kiếm chính xác (hoặc bạn có thể dùng fuzzy matching ở đây)
        if query in title_to_appid:
            appid = title_to_appid[query]
            selected_appids.append(appid)
            print(f"  [OK] Đã thêm: {appid_to_title[appid]}")
        else:
            # Tìm kiếm gần đúng (chứa từ khóa)
            matches = [t for t in title_to_appid.keys() if query in t]
            if matches:
                print(f"  [?] Không tìm thấy chính xác. Ý bạn là:")
                for i, m in enumerate(matches[:5]):
                    print(f"    {i+1}. {appid_to_title[title_to_appid[m]]}")
                print("  Vui lòng nhập lại tên chính xác.")
            else:
                print("  [!] Không tìm thấy game này trong dữ liệu.")

    if not selected_appids:
        print("Bạn chưa chọn game nào!")
        return

    # 2. Folding-in
    # Giảm epochs xuống 100 cho nhanh và tránh overfit khi học 1 user, 
    # vẫn dùng LR và Reg từ Optuna của bạn.
    u_factor, played_indices = fold_in_user(selected_appids, i_factors, i_biases, map_appid_to_idx, epochs=100)
    
    if u_factor is None:
        print("Lỗi khi tính toán sở thích.")
        return

    # 3. Recommend
    recommend(u_factor, i_factors, i_biases, played_indices, map_idx_to_appid, appid_to_title)

if __name__ == '__main__':
    main()
