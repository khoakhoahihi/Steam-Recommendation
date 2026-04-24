import os
import json
import numpy as np
import pandas as pd
import time
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MODEL_DIR = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\model_weights'
METADATA_FILE = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'

def load_model():
    print("[1] Đang tải mô hình VEBPR đã huấn luyện...")
    npz_path = os.path.join(MODEL_DIR, 'vebpr_weights.npz')
    map_path = os.path.join(MODEL_DIR, 'item_map.json')
    
    if not os.path.exists(npz_path) or not os.path.exists(map_path):
        raise FileNotFoundError("Chưa tìm thấy model_weights. Bạn đã chạy export_model.py chưa?")
        
    data = np.load(npz_path)
    i_factors = data['i_factors']
    i_biases = data['i_biases']
    
    with open(map_path, 'r', encoding='utf-8') as f:
        item_map = json.load(f)
        
    return i_factors, i_biases, item_map['idx_to_appid'], item_map['appid_to_idx']

def load_game_titles():
    print("[2] Đang tải thư viện Game Titles...")
    df = pd.read_csv(METADATA_FILE)
    df.reset_index(inplace=True)
    # Pandas đọc cột đầu tiên thành index, và cột thứ hai lọt vào tên 'AppID'
    return {int(app_id): str(name) for app_id, name in zip(df['index'], df['AppID'])}

def mock_get_steam_games(steam_id):
    """
    HÀM MÔ PHỎNG STEAM API: IPlayerService/GetOwnedGames/v1/
    Khi bạn có API key thật, bạn thay đoạn này bằng request.get()
    """
    print(f"\n[API] Đang gọi Steam API lấy dữ liệu cho SteamID: {steam_id}...")
    time.sleep(1) # Giả lập delay mạng
    
    # Giả sử user này chơi các game phổ biến (ví dụ: CS:GO, Dota 2, Left 4 Dead 2)
    # Định dạng trả về: {app_id: playtime_forever_in_minutes}
    mock_data = {
        730: 15000,   # Counter-Strike
        570: 25000,   # Dota 2
        4000: 3000,   # Garry's Mod
        550: 1200,    # Left 4 Dead 2
        105600: 4500, # Terraria
        892970: 5000, # Valheim
    }
    
    return mock_data

def fold_in_user(steam_games, i_factors, i_biases, appid_to_idx, epochs=50, lr=0.01, reg=0.01):
    """
    Phương án B: "Học nhồi" (Folding-In)
    Cố định ma trận i_factors, chỉ cập nhật u_factor cho User mới dựa trên game họ đã chơi.
    """
    print("[3] Bắt đầu Folding-in (Học cấp tốc sở thích người dùng)...")
    
    k = i_factors.shape[1]
    num_items = i_factors.shape[0]
    
    # Lọc ra những game có trong hệ thống
    played_indices = []
    play_weights = []
    
    for app_id, playtime in steam_games.items():
        # steam_games key có thể là int hoặc str tùy vào API, cần ép kiểu
        app_id_int = int(app_id)
        # Chuyển qua string vì key trong json item_map là string
        app_id_str = str(app_id_int)
        
        if app_id_str in appid_to_idx:
            idx = appid_to_idx[app_id_str]
            played_indices.append(idx)
            # Tính trọng số giống như data_loader: 1.0 + log(1 + playtime / 60)
            hours = playtime / 60.0
            weight = 1.0 + np.log1p(hours)
            play_weights.append(weight)
            
    if not played_indices:
        print("[CẢNH BÁO] Không tìm thấy game nào của user này trong cơ sở dữ liệu!")
        return np.zeros(k), []
        
    played_indices = np.array(played_indices, dtype=np.int32)
    play_weights = np.array(play_weights, dtype=np.float32)
    
    # Khởi tạo u_factor ngẫu nhiên giống VEBPR ban đầu
    np.random.seed(42)
    scale = 0.5 / k
    u_factor = np.random.uniform(-scale, scale, size=k).astype(np.float32)
    
    # SGD Loop (bằng Numpy thuần vì code ngắn, chay rất nhanh cho 1 user)
    # Lặp qua các tương tác nhiều lần
    for epoch in range(epochs):
        for idx in range(len(played_indices)):
            i = played_indices[idx]
            w_ui = play_weights[idx]
            
            # Sample negative j
            j = np.random.randint(0, num_items)
            while j in played_indices:
                j = np.random.randint(0, num_items)
                
            # Tính score
            z = u_factor.dot(i_factors[i] - i_factors[j]) + i_biases[i] - i_biases[j]
            
            # Sigmoid (clamp để tránh overflow)
            z_clipped = np.clip(z, -15.0, 15.0)
            sigmoid = 1.0 / (1.0 + np.exp(z_clipped))
            
            grad = w_ui * sigmoid
            
            # Cập nhật u_factor (nhớ trừ đi regularization)
            u_factor += lr * (grad * (i_factors[i] - i_factors[j]) - reg * u_factor)
            
    return u_factor, played_indices

def recommend_for_user(u_factor, i_factors, i_biases, played_indices, idx_to_appid, appid_to_title, N=10):
    print(f"[4] Đang lấy Top-{N} Gợi ý...")
    
    # Tính điểm cho tất cả các item: Score = U * V^T + B
    scores = i_factors.dot(u_factor) + i_biases
    
    # Phạt những game đã chơi để không gợi ý lại
    scores[played_indices] = -np.inf
    
    # Lấy Top N
    top_indices = np.argsort(scores)[::-1][:N]
    
    print("\n" + "="*50)
    print(" KẾT QUẢ GỢI Ý DỰA TRÊN DỮ LIỆU STEAM")
    print("="*50)
    for rank, idx in enumerate(top_indices, 1):
        app_id = int(idx_to_appid[str(idx)])
        title = appid_to_title.get(app_id, f"Unknown Game (AppID: {app_id})")
        score = scores[idx]
        print(f"#{rank:<2} | Điểm: {score:8.4f} | {title} (AppID: {app_id})")

def main():
    # 1. Tải Model & Metadata
    try:
        i_factors, i_biases, appid_to_idx, idx_to_appid = load_model()
    except FileNotFoundError as e:
        print(e)
        return
        
    appid_to_title = load_game_titles()
    
    # 2. Nhập Steam ID (Giả lập)
    # Steam ID 64-bit thường có dạng: 7656119xxxxxxxxxx
    steam_id_input = "76561198012345678"
    
    # 3. Lấy dữ liệu từ Steam (Mock)
    steam_games = mock_get_steam_games(steam_id_input)
    
    print("\n[INFO] Các game User này đã chơi:")
    for app_id, playtime in steam_games.items():
        title = appid_to_title.get(app_id, f"Unknown (AppID: {app_id})")
        print(f"  - {title}: {playtime/60:.1f} giờ")
        
    # 4. Học cấp tốc Sở thích (Folding-In)
    u_factor, played_indices = fold_in_user(steam_games, i_factors, i_biases, appid_to_idx)
    
    if len(played_indices) == 0:
        return
        
    # 5. Gợi ý
    recommend_for_user(u_factor, i_factors, i_biases, played_indices, idx_to_appid, appid_to_title)

if __name__ == '__main__':
    main()
