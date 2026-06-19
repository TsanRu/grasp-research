#!/usr/bin/env python
# -*- coding: utf-8 -*-
# semantic_grasp_analyzer.py

import sys
import os
import warnings

# --- 讓 ROS 可以被找到 (同 brain.py) ---
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    from sensor_msgs.msg import Image
except ImportError:
    print("❌ 找不到 ROS，請確保在 ROS 環境下執行！")
    sys.exit(1)

# --- 強制過濾雜訊 ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
# ---------------------------

import json
import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import pipeline, SamModel, SamProcessor
import google.generativeai as genai

# ⚠️ 使用你的 Key
MY_GEMINI_KEY = "AIzaSyDjvWMftQgnGt3wN_QaKJThZxYaSc6j_0E" 
genai.configure(api_key=MY_GEMINI_KEY)

# ==========================================
# 🧠 系統提示詞
# ==========================================
VISION_SYSTEM_PROMPT = """
你是一個頂尖的雙臂機器人視覺分析專家，專精於安全、自然的物件交接任務。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示左右機械手臂基座、桌面高度、目標物件位置。
【圖片 2】物件特寫網格圖：目標物件的 5x5 網格（X軸 A-E 左到右，Y軸 1-5 上到下），每個格子中心有白色標籤（如 A1、B3）。

【角色定義】
操作臂（右手）：從桌面穩定夾取物件，移動到交接區後透過手腕旋轉調整物件朝向，再遞給左手。
交接臂（左手）：在交接區等待，以自然人類方式從右手接取物件。

【第一步：判斷物件交接策略】
在選擇網格之前，先判斷這個物件屬於哪種類型：

類型一（functional_end）：物件有明確的功能性接取部位
範例：鎚子（握柄）、刀子（刀柄）、水壺（把手）、鍋子（把手）
→ 操作臂夾持非接取端，交接臂接取功能端
→ 例如鎚子：操作臂夾鎚頭，交接臂接握柄

類型二（geometric）：物件無明確功能分區，形狀對稱、規則或不規則
範例：餅乾盒、湯罐、香蕉、書本
→ 操作臂夾持物件中段，交接臂接取夾爪能穩定握持的面

【第二步：根據策略選擇網格，並嚴格遵守以下物理約束】

───── 操作臂（右手）的約束 ─────

約束 R1：高度選擇
Y1 是物件頂部，Y5 是物件底部靠近桌面。
選擇時考量兩件事：
第一，操作臂的夾取位置需要穩定支撐後續的手腕旋轉動作，
中段偏下通常比頂部或底部更穩定。
第二，操作臂夾取的位置應盡量為交接臂保留足夠的接取空間，
避免把整個物件的主要接取面都佔住。
靠近桌面的區域（Y4、Y5）需注意夾爪本體不要撞到桌面，
但 AnyGrasp 本身有碰撞偵測，不需要強制排除這個區域。

約束 R2：物理邊緣必要性
選取的網格必須包含物件真實輪廓（頂部、側面、角落）。
夾爪需要邊緣才能施力，絕不能只選中央平坦區域。

約束 R3：面積與連續性
根據物件實際大小動態決定網格數量（夾爪寬約 7-8cm）：
- 細長型或小物件：2 格通常足夠
- 中型物件（如一般餅乾盒）：2~3 格
- 大型物件：3~4 格
不要因為範例是兩格就固定選兩格，以物件在圖片中的實際比例判斷。
網格必須連續集中，不能分散。

───── 交接臂（左手）的約束 ─────

約束 L1：自然接取方向
先思考「如果是人類要從機器人手中接過這個物件，會自然從哪裡拿？」

functional_end 物件：
直接選功能接取端（握柄、把手）對應的網格區域。

geometric 物件：
優先選擇夾爪能穩定握持的面。判斷方式：
- 規則方形物件（盒子、書本）：優先選窄面或側面，
  大面通常超過夾爪開口（約 8.5cm）難以握持
- 細長或不規則物件（香蕉、湯匙）：
  若窄面不易施力，從上方或側方施力的較大面也是合理選擇，
  不需要強制選窄面，以夾爪能穩定施力為判斷依據
- 圓柱形物件（湯罐）：任何方向皆可，
  選擇交接臂最容易自然接近的那一側

約束 L2：手腕自然性
選擇交接臂手腕不需要大幅扭轉即可自然貼合的區域。

約束 L3：與操作臂的碰撞判斷
判斷方式：想像兩個夾爪同時夾在你選的位置，夾爪本體會不會碰到對方？
- 不會碰到 → 即使網格相鄰也沒問題，不需要強制空出一格
- 會碰到 → 才需要拉開距離
小物件上兩個區域自然接近是正常的，
不要為了製造間距而把某個區域推到不合理位置或物件邊緣外。

約束 L4：物理邊緣必要性
同 R2，選取網格必須包含物件真實邊緣。

約束 L5：面積與連續性
同 R3，至少 2 個相鄰網格，根據物件大小動態調整。

───── 共同約束 ─────

約束 G1：從【圖片 1】確認可達性
觀察物件位置，避免把需要大幅伸展的區域分配給對應手臂。

【輸出格式】（純 JSON，不含任何其他文字或 markdown）
{
    "object_name": "物件英文名稱",
    "handover_strategy": "functional_end 或 geometric",
    "receiver_part": "若為 functional_end 填接取部位名稱，若為 geometric 填 null",
    "left_grids": ["網格代號", ...],   // 數量依物件大小決定，通常 2~4 格
    "right_grids": ["網格代號", ...],  // 數量依物件大小決定，通常 2~4 格
    "reasoning": "1.判斷物件類型的理由 → 2.操作臂選擇這些網格的理由 → 3.交接臂選擇這些網格的理由（符合人類自然接取直覺）→ 4.確認兩區域不碰撞"
}
"""

RECEIVER_ONLY_PROMPT = """
你是一個機器人視覺分析專家，專精於自然的物件交接任務。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示操作臂（右手）夾持物件懸空於交接區域的狀態。
【圖片 2】物件特寫網格圖：目標物件的 5x5 網格（X軸 A-E 左到右，Y軸 1-5 上到下），每個格子中心有白色標籤（如 A1、B3）。

【任務背景】
這是交接任務的第二階段。
操作臂（右手）已夾持物件移動到交接區域並完成旋轉調整，現在靜止等候。
交接臂（左手）需要從空中接取懸空的物件，模擬人類自然接取的動作。
請只為交接臂選擇最適合的接取區域，right_grids 一律回傳空陣列。

【約束】

約束一：迴避操作臂佔據的區域
操作臂已夾持物件中段某個位置，交接臂需要選擇操作臂夾爪不會干涉的區域。
判斷方式：想像交接臂夾爪貼合在你選的位置，與操作臂夾爪會不會碰到對方？
- 不會碰到 → 即使相鄰也沒問題，不需要強制空出一格
- 會碰到 → 才需要移到操作臂未佔據的區域
不要強制規定「一定要選頂部」或「一定要選某個 Y 軸」，
以實際不碰撞為判斷依據，任何位置只要不干涉都是候選。

約束二：自然接取方向
先思考「如果是人類要從機器人手中接過這個物件，會自然從哪裡拿？」
選擇交接臂手腕不需要大幅扭轉即可自然貼合的區域，
從側面、上方、斜向接近都是合理選擇，取決於物件形狀與當前朝向。

約束三：物理邊緣必要性
選取網格必須包含物件真實邊緣（頂部、左側、右側或角落輪廓）。
夾爪需要邊緣才能施力，不能只選中央平坦區域。

約束四：面積與連續性
根據物件實際大小動態決定網格數量（夾爪寬約 7-8cm）：
- 細長型或小物件：2 格通常足夠
- 中型物件：2~3 格
- 大型物件：3~4 格
不要因為輸出範例是兩格就固定選兩格，以物件在圖片中的實際比例判斷。
網格必須連續集中，不能分散。

約束五：左手臂可達性
從【圖片 1】確認物件在交接區域的位置，
選擇左手能自然到達、不需要大幅伸展或扭轉手臂的區域。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "left_grids": ["網格代號", ...],   // 數量依物件大小決定，通常 2~4 格
    "right_grids": [],
    "reasoning": "1.觀察操作臂夾持位置在哪裡 → 2.判斷哪些區域可以不干涉地接取 → 3.從中選擇最符合人類自然接取直覺的位置 → 4.確認面積與邊緣條件"
}
"""

def imgmsg_to_numpy(msg):
    """將 ROS Image 轉為 Numpy Array (RGB)"""
    dtype_class = np.uint8
    channels = 3 if "rgb8" in msg.encoding or "bgr8" in msg.encoding else 1
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
        if "bgr8" in msg.encoding:
            img = img[:, :, ::-1] # 轉為 RGB
    else:
        img = img.reshape((msg.height, msg.width))
    return img

def draw_som_grid(img_rgb, rows=5, cols=5):
    """改良版 Set-of-Mark 網格繪製"""
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]

    overlay = img_rgb.copy()
    alpha = 0.15
    colors = [(173, 216, 230), (255, 200, 150)]
    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            grid_id = f"{col_labels[c]}{r + 1}"
            grid_dict[grid_id] = (x1, y1, x2, y2)
            color = colors[(r + c) % 2]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    result = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(result, (0, y), (w, y), (80, 80, 80), 1)
    for j in range(1, cols):
        x = int(j * cell_w)
        cv2.line(result, (x, 0), (x, h), (80, 80, 80), 1)

    cv2.rectangle(result, (0, 0), (w - 1, h - 1), (80, 80, 80), 2)

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * cell_w), int(r * cell_h)
            x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
            grid_id = f"{col_labels[c]}{r + 1}"
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(cell_w, cell_h) / 60.0
            thickness = max(1, int(font_scale * 2))
            
            (text_w, text_h), _ = cv2.getTextSize(grid_id, font, font_scale, thickness)
            text_x, text_y = cx - text_w // 2, cy + text_h // 2
            cv2.putText(result, grid_id, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(result, grid_id, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return result, grid_dict

class SemanticPromptTester:
    def __init__(self):
        rospy.init_node('semantic_prompt_tester', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📦 正在載入模型 ({self.device})... 這可能需要一點時間")
        
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        
        self.save_dir = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"
        os.makedirs(self.save_dir, exist_ok=True)
        print("✅ 模型載入完成！")

    def run_test(self, object_name, mode="dual"):
        print(f"\n⏳ 正在等待相機畫面 (Topic: /camera/color/image_raw) ...")
        try:
            # ✅ 直接在這裡抓取一張最新畫面，不需要寫 Subscriber 等 trigger
            msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10.0)
            print("📸 成功擷取到相機最新實況畫面！")
        except rospy.ROSException:
            print("❌ 逾時！無法接收到相機畫面，請檢查 Gazebo 是否有在跑？")
            return

        try:
            print(f"\n🚀 開始處理管線，目標: {object_name}，模式: {mode}")
            
            # 1. 影像轉換與儲存 (第一步！)
            img_np = imgmsg_to_numpy(msg) # 這裡是 RGB
            h, w = img_np.shape[:2]
            img_pil = PILImage.fromarray(img_np)
            
            # 儲存到 Gazebo 資料夾供後續檢查 (轉回 BGR 存檔)
            cv2.imwrite(os.path.join(self.save_dir, "original_rgb.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            print("✅ 第一步：已成功儲存實況相機畫面 original_rgb.png")

            # 2. OWL-v2 偵測
            print(f"🦉 [1/4] OWL-v2 尋找 '{object_name}'...")
            preds = self.detector(img_pil, candidate_labels=[object_name])
            if not preds:
                print(f"❌ OWL-v2 找不到 '{object_name}'")
                return

            best_pred = max(preds, key=lambda x: x['score'])
            box = best_pred['box']
            x_min, y_min, x_max, y_max = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
            print(f"   偵測到 '{object_name}'，信心度: {best_pred['score']:.2f}")

            # 裁切物件區域
            pad = 20
            c_xmin, c_ymin = max(0, x_min - pad), max(0, y_min - pad)
            c_xmax, c_ymax = min(w, x_max + pad), min(h, y_max + pad)
            cropped_img = img_np[c_ymin:c_ymax, c_xmin:c_xmax].copy() # 裁切 RGB
            cv2.imwrite(os.path.join(self.save_dir, "object_crop_raw.png"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

            # 3. 改良版 SoM 網格繪製
            print("✂️ [2/4] 繪製 Set-of-Mark 網格...")
            grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

            grid_dict_absolute = {}
            for grid_id, (lx1, ly1, lx2, ly2) in grid_dict_local.items():
                grid_dict_absolute[grid_id] = [c_xmin + lx1, c_ymin + ly1, c_xmin + lx2, c_ymin + ly2]

            grid_img_path = os.path.join(self.save_dir, "cropped_grid_for_vlm.png")
            cv2.imwrite(grid_img_path, cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR))

            # 4. Gemini 推理
            print("🧠 [3/4] 呼叫 Gemini 進行 Prompt 測試...")
            gemini_local_img = PILImage.open(grid_img_path)
            if mode == "receiver_only":
                prompt = RECEIVER_ONLY_PROMPT
            else:
                prompt = VISION_SYSTEM_PROMPT
            
            try:
                token_info = self.gemini_model.count_tokens([prompt, img_pil, gemini_local_img])
                print(f"📊 [Token 預估] 這次請求將消耗 Input Token: {token_info.total_tokens}")
            except Exception as e:
                print(f"⚠️ 無法計算 Token: {e}")
    
            response = self.gemini_model.generate_content([prompt, img_pil, gemini_local_img])
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            vlm_result = json.loads(clean_json)

            receiver_grids = vlm_result.get('left_grids', [])
            giver_grids = vlm_result.get('right_grids', [])
            print(f"\n💡 Gemini 輸出結果:")
            print(json.dumps(vlm_result, ensure_ascii=False, indent=4))
            print("\n")

            if mode == "left_only":
                giver_grids = []

            # 5. SAM 分割
            print("✂️ [4/4] SAM 精準切割...")
            inputs = self.sam_processor(
                img_pil,
                input_boxes=[[[x_min, y_min, x_max, y_max]]],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs.original_sizes.cpu(),
                inputs.reshaped_input_sizes.cpu()
            )
            global_mask = masks[0][0][0].numpy()
            cv2.imwrite(os.path.join(self.save_dir, "sam_global_mask_full.png"), (global_mask * 255).astype(np.uint8))

            def save_final_mask(grid_ids_list, filename):
                final_mask = np.zeros_like(global_mask, dtype=bool)
                for grid_id in grid_ids_list:
                    if grid_id in grid_dict_absolute:
                        gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                        final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]
                cv2.imwrite(os.path.join(self.save_dir, filename), (final_mask * 255).astype(np.uint8))
                print(f"✅ 遮罩已儲存: {filename}")

            if receiver_grids: save_final_mask(receiver_grids, "receiver_mask.png")
            if giver_grids: save_final_mask(giver_grids, "giver_mask.png")

            print("\n🎉 測試管線順利執行完畢！")

        except Exception as e:
            print(f"❌ 管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # 🟢 你要改目標物件或模式，直接改下面這兩行就好！
    TARGET_OBJECT = "hammer"
    TEST_MODE = "dual" # "dual" 或 "left_only"
    
    print("🚀 啟動相機直連測試模式...")
    tester = SemanticPromptTester()
    tester.run_test(TARGET_OBJECT, mode=TEST_MODE)