import cv2
import numpy as np
import os

def overlay_masks(base_image_path, receiver_mask_path, giver_mask_path, output_path):
    print("🎨 開始進行遮罩疊加處理...")
    
    # 1. 讀取影像
    base_img = cv2.imread(base_image_path)
    # 遮罩讀取為單通道灰階圖
    receiver_mask = cv2.imread(receiver_mask_path, cv2.IMREAD_GRAYSCALE)
    giver_mask = cv2.imread(giver_mask_path, cv2.IMREAD_GRAYSCALE)

    if base_img is None or receiver_mask is None or giver_mask is None:
        print("❌ 找不到圖片檔案，請確認路徑是否正確！")
        return

    # 2. 尺寸安全檢查 (處理全局遮罩配局部小圖的狀況)
    h, w = base_img.shape[:2]
    if receiver_mask.shape[:2] != (h, w):
        print("⚠️ 警告：遮罩與底圖尺寸不符，正在自動縮放遮罩對齊底圖...")
        receiver_mask = cv2.resize(receiver_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        giver_mask = cv2.resize(giver_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. 建立彩色疊加層
    # 複製一張底圖作為畫布
    overlay = base_img.copy()

    # OpenCV 的顏色通道是 BGR (藍, 綠, 紅)
    # 將 receiver (左手) 標示為【半透明藍色】
    # 條件：當遮罩像素值大於 0 (白色區域) 時，把 overlay 該位置塗成藍色
    overlay[receiver_mask > 0] = [255, 100, 100]  # 淡藍色

    # 將 giver (右手) 標示為【半透明橘紅色】
    overlay[giver_mask > 0] = [50, 100, 255]   # 橘紅色

    # 4. 執行 Alpha Blending 半透明混合
    alpha = 1.0  # 遮罩顏色的不透明度 (0.0 完全透明, 1.0 完全不透明)
    
    # cv2.addWeighted 公式: dst = src1*alpha + src2*beta + gamma
    result = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)

    # 5. 儲存結果
    cv2.imwrite(output_path, result)
    print(f"✅ 疊加完成！結果已儲存至: {output_path}")

if __name__ == "__main__":
    # --- 這裡替換成你的實際檔案路徑 ---
    # 💡 強烈建議這裡使用 original_rgb.png 來獲得最精準的物理位置對齊
    BASE_IMG = "my_gazebo_data/original_rgb.png" 
    RECEIVER = "my_gazebo_data/receiver_mask.png"
    GIVER = "my_gazebo_data/giver_mask.png"
    OUTPUT = "my_gazebo_data/visualized_grasp_result.png"

    overlay_masks(BASE_IMG, RECEIVER, GIVER, OUTPUT)