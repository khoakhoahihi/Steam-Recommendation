import pandas as pd
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Đường dẫn
ORIGINAL_GAMES = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games.csv'
LITE_GAMES = r'C:\Users\Lenovo\PycharmProjects\ShopeeRanking\data\games_lite.csv'

def create_lite_version():
    print("Đang tạo bản rút gọn của games.csv để upload GitHub...")
    # Chỉ đọc những cột cần thiết
    # Do lỗi lệch cột, ta lấy 3 cột đầu (thực tế là AppID, Name) và cột Header image
    df = pd.read_csv(ORIGINAL_GAMES)
    df.reset_index(inplace=True)
    
    # Lọc lấy AppID (index), Name (AppID), và Image (Header image)
    df_lite = df[['index', 'AppID', 'Header image']].copy()
    df_lite.columns = ['AppID', 'Name', 'Image']
    
    # Xóa các dòng lỗi
    df_lite = df_lite.dropna(subset=['AppID', 'Name'])
    
    # Lưu lại bản nhẹ
    df_lite.to_csv(LITE_GAMES, index=False)
    print(f"Hoàn thành! File mới: {LITE_GAMES}")
    print(f"Kích thước giảm từ ~400MB xuống còn khoảng {os.path.getsize(LITE_GAMES)/1024/1024:.2f} MB")

if __name__ == '__main__':
    create_lite_version()
