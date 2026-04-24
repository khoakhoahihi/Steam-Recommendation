import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import sys

# Cấu hình trang
st.set_page_config(
    page_title="Steam AI Recommender",
    page_icon="🎮",
    layout="wide"
)

# Đường dẫn file (Dùng relative path để chạy được trên GitHub/Streamlit Cloud)
MODEL_DIR = 'model_weights'
METADATA_FILE = os.path.join('data', 'games_lite.csv')

# 1. Hàm tải Model (Cần Cache để không load lại mỗi lần nhấn nút)
@st.cache_resource
def load_model():
    npz_path = os.path.join(MODEL_DIR, 'vebpr_weights.npz')
    map_path = os.path.join(MODEL_DIR, 'item_map.json')
    
    if not os.path.exists(npz_path) or not os.path.exists(map_path):
        return None, None, None, None
        
    data = np.load(npz_path)
    i_factors = data['i_factors']
    i_biases = data['i_biases']
    
    with open(map_path, 'r', encoding='utf-8') as f:
        item_map = json.load(f)
        
    # item_map['idx_to_appid'] chứa {appid: idx}
    # item_map['appid_to_idx'] chứa {idx: appid}
    return i_factors, i_biases, item_map['idx_to_appid'], item_map['appid_to_idx']

# 2. Hàm tải Database Game
@st.cache_data
def load_game_db():
    df = pd.read_csv(METADATA_FILE)
    
    appid_to_title = {int(row['AppID']): str(row['Name']) for _, row in df.iterrows()}
    appid_to_image = {int(row['AppID']): str(row['Image']) for _, row in df.iterrows() if pd.notna(row['Image'])}
    
    game_options = [f"{row['Name']} (ID: {int(row['AppID'])})" for _, row in df.iterrows()]
    title_to_appid = {f"{row['Name']} (ID: {int(row['AppID'])})": int(row['AppID']) for _, row in df.iterrows()}
    
    return appid_to_title, game_options, title_to_appid, appid_to_image

# 3. Thuật toán Folding-in (SGD cho User mới)
def fold_in_user(selected_appids, i_factors, i_biases, appid_to_idx, epochs=50):
    k = i_factors.shape[1]
    num_items = i_factors.shape[0]
    
    played_indices = []
    for app_id in selected_appids:
        app_id_str = str(app_id)
        if app_id_str in appid_to_idx:
            played_indices.append(appid_to_idx[app_id_str])
            
    if not played_indices:
        return None, []
        
    played_indices = np.array(played_indices, dtype=np.int32)
    np.random.seed(42)
    u_factor = np.random.uniform(-0.5/k, 0.5/k, size=k).astype(np.float32)
    
    lr = 0.02
    reg = 0.01
    
    for _ in range(epochs):
        for i in played_indices:
            j = np.random.randint(0, num_items)
            while j in played_indices:
                j = np.random.randint(0, num_items)
            
            z = u_factor.dot(i_factors[i] - i_factors[j]) + i_biases[i] - i_biases[j]
            sigmoid = 1.0 / (1.0 + np.exp(np.clip(z, -15, 15)))
            u_factor += lr * (3.0 * sigmoid * (i_factors[i] - i_factors[j]) - reg * u_factor)
            
    return u_factor, played_indices

# --- GIAO DIỆN STREAMLIT ---

st.title("Steam AI Game Recommender")
st.markdown("""
Hệ thống sử dụng mô hình **VEBPR (View-Enhanced BPR)** để thấu hiểu gu chơi game của bạn.
Hãy chọn những game bạn tâm đắc nhất, AI sẽ tìm ra những game dành riêng cho bạn!
""")

# Load data
i_factors, i_biases, map_appid_to_idx, map_idx_to_appid = load_model()
appid_to_title, game_options, title_to_appid, appid_to_image = load_game_db()

if i_factors is None:
    st.error("Không tìm thấy dữ liệu Model. Vui lòng chạy export_model.py trước!")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Nhập sở thích của bạn")
    selected_names = st.multiselect(
        "Chọn các game bạn thích nhất:",
        options=game_options,
        placeholder="Ví dụ: Elden Ring, Dota 2...",
        help="Càng chọn nhiều game, gợi ý càng chính xác!"
    )
    
    num_rec = st.slider("Số lượng gợi ý:", 5, 20, 10)
    
    btn = st.button(" Gợi ý ngay", type="primary", use_container_width=True)

with col2:
    if btn:
        if not selected_names:
            st.warning("Vui lòng chọn ít nhất 1 game!")
        else:
            with st.spinner("Đang phân tích gu của bạn..."):
                # Lấy AppIDs
                selected_appids = [title_to_appid[name] for name in selected_names]
                
                # Folding in
                u_factor, played_indices = fold_in_user(selected_appids, i_factors, i_biases, map_appid_to_idx)
                
                if u_factor is not None:
                    # Calculate scores
                    scores = i_factors.dot(u_factor) + i_biases
                    scores[played_indices] = -np.inf
                    top_indices = np.argsort(scores)[::-1][:num_rec]
                    
                    st.subheader(f"Top {num_rec} Game dành riêng cho bạn:")
                    
                    for rank, idx in enumerate(top_indices, 1):
                        app_id = int(map_idx_to_appid[str(idx)])
                        title = appid_to_title.get(app_id, f"AppID: {app_id}")
                        img_url = appid_to_image.get(app_id)
                        
                        # Hiển thị dạng Card
                        with st.container(border=True):
                            c_img, c_txt = st.columns([0.3, 0.7])
                            if img_url:
                                c_img.image(img_url, use_container_width=True)
                            else:
                                c_img.write("No Image")
                                
                            c_txt.write(f"### {rank}. {title}")
                            c_txt.caption(f"Steam AppID: {app_id}")
                else:
                    st.error("Dữ liệu game bạn chọn chưa được hỗ trợ trong model này.")
    else:
        st.info("Hãy chọn game ở bên trái và nhấn nút 'Gợi ý ngay' để bắt đầu.")

# Footer
st.divider()
st.caption("Built with ❤️ by DangKhoa and Antigravity | Powered by VEBPR Algorithm")
