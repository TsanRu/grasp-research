#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    print("✅ 成功跨界連接 ROS Noetic！")
except ImportError:
    print("❌ 找不到 ROS")

import cv2
import json
import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage
from transformers import pipeline, SamModel, SamProcessor
import google.generativeai as genai
import warnings
from sensor_msgs.msg import JointState

warnings.filterwarnings('ignore')

# MY_GEMINI_KEY = "AIzaSyBrJiv3YmFDBG5eWdBrUfkxOOUq7VUtMRU"
MY_GEMINI_KEY = "AIzaSyAMnR6DutgMRXnHoSNCwoEjMwq-46n1VYo"

genai.configure(api_key=MY_GEMINI_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 術語定義：
#   接收臂 (receiver arm) = 左臂，末端裝有 Hand-in-Eye 相機
#   操作臂 (operator arm) = 右臂，負責從桌面夾取並搬運物件
# ─────────────────────────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """
你是一個頂尖的雙臂機器人視覺分析專家。

你將收到兩張圖片：
【圖片 1】全局場景圖：顯示左右機械手臂基座位置、桌面高度，以及目標物件的真實擺放狀態。
【圖片 2】物件特寫網格圖：目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。
每個格子的中心都有白色標籤顯示其代號（如 A1、B2）。

【任務背景】
你看到的影像來自安裝在接收臂（左臂）末端的相機視角。
操作臂（右臂）負責從桌面抓起物件並舉起，接收臂在空中從側面或上方接過物件完成交接。
由於相機在接收臂上，畫面中接收臂本身可能部分可見或被裁切，請以物件為判斷中心。

【做決策時必須考慮的物理約束，全部同等重要】

約束一：高度選擇原則
Y1 是物件頂部，Y5 是物件底部靠近桌面。
網格的 Y 軸代表高度，但請【絕對不要死記網格編號】來決定安全高度！
實體機械夾爪的手指有其厚度，若在貼近桌面的位置下爪，夾爪會直接撞擊桌面導致任務失敗。
通常各高度區間的可用性如下（具體仍請根據物件及環境去考量）：
- Y1：適合接收臂從上方接取，不適合操作臂（會佔據上方空間導致接收臂無法進入）
- Y2、Y3：最佳抓取區間，手臂工作空間充裕，強烈推薦優先選擇
- Y4：可以使用，手臂需要稍微向下延伸，但仍在合理範圍內
- Y5：盡量避免，非常靠近桌面，手臂難以到達且容易碰撞桌面

請注意：Y2、Y3、Y4 都是可以使用的區間，不要因為「靠近桌面」就過度迴避 Y3 和 Y4。
物件若較矮，Y2 可能就已經是中段偏下，此時 Y3 仍應考慮使用。
目標是選擇能穩定夾取的最高可用位置，而不是只選最高排。

約束二：兩個抓取區域需有足夠但不過度的分離
目標是「剛好夠用的間距」，而不是「最大化距離」。

判斷方式：想像兩個夾爪同時夾在你選的位置，它們會不會碰到對方？
- 兩個區域相鄰（只差一格），若夾爪很可能會干涉，再拉開距離；若夾爪空間足夠，則相鄰也無妨
- 不要為了拉開距離而把某隻手推到物件邊緣或角落
- 間隔可以是 X 軸方向（左右拉開）、Y 軸方向（上下拉開），或對角方向，取決於物件形狀和你選的分工策略

請根據物件的實際形狀和兩隻手臂的位置，自由選擇最合理的分工方式：
操作臂與接收臂的左右分工、上下分工、對角分工都是允許的，選擇讓兩隻手臂都能自然接近物件的方案。
不要因為要滿足間距而把某隻手的區域推到不合理的高度（例如被迫選 Y1 或 Y5）。

約束三：依據視覺比例動態決定抓取面積
不要拘泥於固定的網格數量。請觀察【圖片 2】中物件的實際大小與網格的比例。
真實機械夾爪約有 7~8 公分寬，你選取的網格總面積必須足以容納一個實體夾爪的貼合面：
- 若物件很小或屬於細長型，2 個網格可能就已足夠。
- 若物件較大，則可能需要 3 到 4 個網格。
請靈活給予「剛好能讓夾爪穩定貼合且連續」的網格範圍，避免面積過小導致物理引擎算不出姿態，也避免面積過大失去抓取精準度。

約束四：每個抓取區域需連續且集中
每隻手分配到的網格應彼此相鄰，分散的網格無法形成有效的夾取面。
每隻手至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束五：從【圖片 1】判斷手臂可達性
觀察物件在桌面的實際位置，若物件偏向某一側導致該側手臂需要大幅伸展，則避免將難度高的區域分配給該側手臂。

約束六：物理邊緣必要性
兩指平行的機械夾爪無法抓取平坦的表面，必須依賴物件的「實體邊緣」或「角落」才能施力。
因此，你挑選的網格區域群【必須包含物件的真實邊緣】（例如頂部、左側或右側輪廓）。
最好的做法是挑選「包含邊緣及其相鄰內部區域」的網格群，
這樣夾爪既能摸到邊緣，又有足夠的平面可以貼合受力。絕對不能只挑選位於物件正中央的平坦網格。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "receiver_grids": ["網格代號", ...],
    "operator_grids": ["網格代號", ...],
    "reasoning": "說明你如何根據上述約束做出這個分配決策"
}
"""

RECEIVER_PROMPT = """
你是一個頂尖的機器人視覺分析專家。

你將收到一張圖片：
【圖片】物件特寫網格圖：目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。
每個格子的中心都有白色標籤顯示其代號（如 A1、B2）。

【任務背景】
這是任務的第二階段。操作臂（右臂）已夾持物件移動到交接區域並靜止等候。
影像來自安裝在接收臂（左臂）末端的相機，接收臂靠近物件準備從操作臂接取。
操作臂的夾取位置已固定在物件中段偏下區域，物件頂部區域是接收臂可以進入接取的空間。
請只為接收臂選擇最佳接取區域，operator_grids 一律回傳空陣列。

【做決策時必須考慮的物理約束，全部同等重要】

約束一：高度選擇原則
Y1 是物件頂部，Y5 是物件底部。
接收臂從上方或側面接取懸空中的物件，與桌面碰撞的風險不存在，但手臂仍有工作空間限制。
通常各高度區間的可用性如下：
- Y1、Y2：物件頂部，接收臂從上方接取的最佳區間，優先考慮
- Y3：中段，可用，但需確認不與操作臂夾取位置干涉
- Y4、Y5：物件下半部，操作臂正在夾持的區域附近，接收臂進入容易與操作臂夾爪干涉，盡量避免

請注意：接收臂的任務是從頂部或側面接取，選擇的位置應讓兩手夾爪不會碰撞。
不要因為「Y1 不常用」就迴避頂部區間，在接取情境下 Y1、Y2 反而是最合理的選擇。

約束二：與操作臂夾取區域保持足夠間距
操作臂已固定夾住物件的中段偏下位置，接收臂必須選擇與操作臂夾取位置有足夠間距的區域。
判斷方式：想像兩個夾爪同時夾在你選的位置，它們會不會碰到對方？
- 若接收臂選擇的區域與操作臂位置相鄰，夾爪很可能干涉，需要往頂部方向移動
- 間距可以是 Y 軸方向（上下拉開）或對角方向，取決於物件形狀
不要為了拉開距離而把接收臂推到完全不合理的位置。

約束三：依據視覺比例動態決定抓取面積
不要拘泥於固定的網格數量。請觀察圖片中物件的實際大小與網格的比例。
真實機械夾爪約有 7~8 公分寬，你選取的網格總面積必須足以容納一個實體夾爪的貼合面：
- 若物件很小或屬於細長型，2 個網格可能就已足夠
- 若物件較大，則可能需要 3 到 4 個網格
請靈活給予「剛好能讓夾爪穩定貼合且連續」的網格範圍。

約束四：抓取區域需連續且集中
選擇的網格應彼此相鄰，分散的網格無法形成有效的夾取面。
至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束五：物理邊緣必要性
兩指平行的機械夾爪無法抓取平坦的表面，必須依賴物件的「實體邊緣」或「角落」才能施力。
你挑選的網格區域群【必須包含物件的真實邊緣】（例如頂部、左側或右側輪廓）。
最好的做法是挑選「包含邊緣及其相鄰內部區域」的網格群。
絕對不能只挑選位於物件正中央的平坦網格。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "receiver_grids": ["網格代號", ...],
    "operator_grids": [],
    "reasoning": "說明你如何根據上述約束做出這個選擇"
}
"""

OPERATOR_PROMPT = """
你是一個頂尖的機器人視覺分析專家。

你將收到一張圖片：
【圖片】物件特寫網格圖：目標物件的局部放大裁切圖，疊加了 5x5 網格（X 軸 A-E 由左到右，Y 軸 1-5 由上到下）。
每個格子的中心都有白色標籤顯示其代號（如 A1、B2）。

【任務背景】
這是任務的第一階段。影像來自安裝在接收臂（左臂）末端的相機，接收臂目前靜止於觀測位置。
操作臂（右臂）需要從桌面夾起物件，移動到交接區域後靜止等候。
接收臂的接取姿態將在操作臂到位後重新感知決定，現在只需要為操作臂規劃初始夾取。
receiver_grids 一律回傳空陣列。

【做決策時必須考慮的物理約束，全部同等重要】

約束一：高度選擇原則
Y1 是物件頂部，Y5 是物件底部靠近桌面。
網格的 Y 軸代表高度，但請【絕對不要死記網格編號】來決定安全高度！
實體機械夾爪的手指有其厚度，若在貼近桌面的位置下爪，夾爪會直接撞擊桌面導致任務失敗。
通常各高度區間的可用性如下（具體仍請根據物件及環境去考量）：
- Y1：物件頂部，操作臂從上方夾取可能導致佔據頂部空間，使後續接收臂接取困難，盡量避免
- Y2、Y3：最佳夾取區間，手臂工作空間充裕，強烈推薦優先選擇
- Y4：可以使用，手臂需要稍微向下延伸，但仍在合理範圍內
- Y5：盡量避免，非常靠近桌面，夾爪容易撞擊桌面

請注意：Y2、Y3、Y4 都是可以使用的區間，不要因為「靠近桌面」就過度迴避 Y3 和 Y4。
物件若較矮，Y2 可能就已經是中段偏下，此時 Y3 仍應考慮使用。
目標是選擇能穩定夾取且不阻礙後續接收臂接取的位置。

約束二：為後續接收臂接取保留空間
操作臂夾取後物件會被舉起懸空，接收臂需要從旁接取。
操作臂應避免夾取物件頂部（Y1 區間），以免佔據接收臂接取時需要進入的空間。
盡量選擇物件中段偏下的位置，讓物件頂部保留給接收臂。

約束三：依據視覺比例動態決定抓取面積
不要拘泥於固定的網格數量。請觀察圖片中物件的實際大小與網格的比例。
真實機械夾爪約有 7~8 公分寬，你選取的網格總面積必須足以容納一個實體夾爪的貼合面：
- 若物件很小或屬於細長型，2 個網格可能就已足夠
- 若物件較大，則可能需要 3 到 4 個網格
請靈活給予「剛好能讓夾爪穩定貼合且連續」的網格範圍，避免面積過小導致物理引擎算不出姿態，也避免面積過大失去抓取精準度。

約束四：抓取區域需連續且集中
選擇的網格應彼此相鄰，分散的網格無法形成有效的夾取面。
至少選擇 2 個相鄰網格，單一網格夾取面積不足。

約束五：物理邊緣必要性
兩指平行的機械夾爪無法抓取平坦的表面，必須依賴物件的「實體邊緣」或「角落」才能施力。
你挑選的網格區域群【必須包含物件的真實邊緣】（例如頂部、左側或右側輪廓）。
最好的做法是挑選「包含邊緣及其相鄰內部區域」的網格群，
這樣夾爪既能摸到邊緣，又有足夠的平面可以貼合受力。絕對不能只挑選位於物件正中央的平坦網格。

【輸出格式】（純 JSON，不含其他文字）
{
    "object_name": "物件英文名稱",
    "receiver_grids": [],
    "operator_grids": ["網格代號", ...],
    "reasoning": "說明你如何根據上述約束做出這個選擇"
}
"""

def imgmsg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 3 if "rgb8" in msg.encoding or "bgr8" in msg.encoding else 1
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
        if "bgr8" in msg.encoding:
            img = img[:, :, ::-1]
    else:
        img = img.reshape((msg.height, msg.width))
    return img


def draw_som_grid(img_rgb, rows=5, cols=5):
    """
    改良版 Set-of-Mark 網格繪製
    在每個格子中心放代號，用半透明色塊交替標示格子
    回傳：標注後的影像、每個格子的絕對座標字典
    """
    h, w = img_rgb.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    col_labels = [chr(65 + i) for i in range(cols)]

    overlay = img_rgb.copy()
    alpha = 0.15

    colors = [
        (173, 216, 230),  # 淡藍
        (255, 200, 150),  # 淡橘
    ]

    grid_dict = {}

    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

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
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            grid_id = f"{col_labels[c]}{r + 1}"
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = min(cell_w, cell_h) / 60.0
            thickness = max(1, int(font_scale * 2))

            (text_w, text_h), _ = cv2.getTextSize(grid_id, font, font_scale, thickness)
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2

            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(result, grid_id, (text_x, text_y),
                        font, font_scale, (255, 255, 255), thickness)

    return result, grid_dict


class SemanticBrainNode:
    def __init__(self):
        rospy.init_node('semantic_brain_node', anonymous=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        rospy.loginfo(f"📦 正在載入模型 ({self.device})...")
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble",
            task="zero-shot-object-detection",
            device=self.device
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.gemini_model = genai.GenerativeModel('gemini-flash-latest')
        rospy.loginfo("✅ 所有 AI 模型載入完成！")

        self.save_dir = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"
        self.target_object = ""
        self.need_process = False
        self.latest_image = None
        self.is_processing = False

        self.arm_is_stable = False
        self.last_joint_positions = None
        self.stable_threshold = 0.001
        self.stable_check_duration = 0.5
        self.stable_since = None

        self.joint_sub = rospy.Subscriber(
            '/joint_states', JointState, self.joint_state_callback)

        self.image_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.image_buffer_callback)

        rospy.Subscriber("/system/trigger_llm", String, self.trigger_callback)

        self.done_pub = rospy.Publisher("/system/llm_done", String, queue_size=1)

        rospy.loginfo("🧠 大腦節點就緒，等待 trigger_llm 訊號...")

    def image_buffer_callback(self, msg):
        """持續接收最新影像，不做處理"""
        self.latest_image = msg

    def trigger_callback(self, msg):
        """收到手臂控制端的觸發訊號"""
        if self.is_processing:
            rospy.logwarn("⏳ AI 正在處理中，忽略重複的 trigger 訊號...")
            return

        try:
            data = json.loads(msg.data)
            self.target_object = data.get("object_name", "unknown")
            self.mode = data.get("mode", "operator_only")  # "dual" 或 "receiver_only"
        except json.JSONDecodeError:
            self.target_object = msg.data
            self.mode = "dual"

        rospy.loginfo(f"⚡ 收到 trigger_llm，目標: {self.target_object}，模式: {self.mode}")
        self.need_process = True

    def joint_state_callback(self, msg):
        """監測接收臂關節是否靜止（使用 ROS joint 名稱前綴 'leftarm_'）"""
        left_indices = [i for i, name in enumerate(msg.name)
                        if name.startswith('leftarm_')]
        current = np.array([msg.position[i] for i in left_indices])

        if self.last_joint_positions is None:
            self.last_joint_positions = current
            return

        delta = np.max(np.abs(current - self.last_joint_positions))
        self.last_joint_positions = current

        if delta < self.stable_threshold:
            if self.stable_since is None:
                self.stable_since = rospy.Time.now()
            elif (rospy.Time.now() - self.stable_since).to_sec() > self.stable_check_duration:
                self.arm_is_stable = True
        else:
            self.arm_is_stable = False
            self.stable_since = None

    def run(self):
        """主迴圈，檢查是否需要處理"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.need_process and self.latest_image is not None:
                if not self.arm_is_stable:
                    rospy.logwarn_throttle(1.0, "⏳ 等待接收臂穩定...")
                    rate.sleep()
                    continue
                self.need_process = False
                self.is_processing = True
                self.process(self.latest_image, self.target_object, self.mode)
                self.is_processing = False
            rate.sleep()

    def process(self, img_msg, object_name, mode="dual"):
        rospy.loginfo(f"📸 開始處理影像，物件: {object_name}，模式: {mode}")
        try:
            img_np = imgmsg_to_numpy(img_msg)
            h, w = img_np.shape[:2]
            img_pil = PILImage.fromarray(img_np)

            cv2.imwrite(
                os.path.join(self.save_dir, "original_rgb.png"),
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            )

            rospy.loginfo(f"🦉 [1/4] OWL-v2 尋找 '{object_name}'...")
            preds = self.detector(img_pil, candidate_labels=[object_name])
            if not preds:
                rospy.logerr(f"❌ 找不到 '{object_name}'")
                self.done_pub.publish(json.dumps({"status": "fail", "reason": "object_not_found"}))
                return

            best_pred = max(preds, key=lambda x: x['score'])
            box = best_pred['box']
            x_min = int(box['xmin'])
            y_min = int(box['ymin'])
            x_max = int(box['xmax'])
            y_max = int(box['ymax'])
            rospy.loginfo(f"   偵測到 '{object_name}'，信心度: {best_pred['score']:.2f}")

            pad = 20
            c_xmin = max(0, x_min - pad)
            c_ymin = max(0, y_min - pad)
            c_xmax = min(w, x_max + pad)
            c_ymax = min(h, y_max + pad)
            cropped_img = img_np[c_ymin:c_ymax, c_xmin:c_xmax].copy()

            cv2.imwrite(
                os.path.join(self.save_dir, "object_crop_raw.png"),
                cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            )

            rospy.loginfo("✂️ [2/4] 繪製 Set-of-Mark 網格...")
            grid_img_rgb, grid_dict_local = draw_som_grid(cropped_img, rows=5, cols=5)

            grid_dict_absolute = {}
            for grid_id, (lx1, ly1, lx2, ly2) in grid_dict_local.items():
                grid_dict_absolute[grid_id] = [
                    c_xmin + lx1, c_ymin + ly1,
                    c_xmin + lx2, c_ymin + ly2
                ]

            grid_img_path = os.path.join(self.save_dir, "cropped_grid_for_vlm.png")
            cv2.imwrite(grid_img_path, cv2.cvtColor(grid_img_rgb, cv2.COLOR_RGB2BGR))

            rospy.loginfo("🧠 [3/4] 呼叫 Gemini 分析網格...")
            gemini_local_img = PILImage.open(grid_img_path)
            if mode == "operator_only":
                prompt = OPERATOR_PROMPT
            elif mode == "receiver_only":
                prompt = RECEIVER_PROMPT
            else:
                rospy.logwarn(f"⚠️ 未知模式 '{mode}'，預設使用 operator_only")
                prompt = OPERATOR_PROMPT

            try:
                token_info = self.gemini_model.count_tokens([prompt, gemini_local_img])
                rospy.loginfo(f"📊 [Token 預估] 這次請求將消耗 Input Token: {token_info.total_tokens}")
            except Exception as e:
                rospy.logwarn(f"⚠️ 無法計算 Token: {e}")

            response = self.gemini_model.generate_content([prompt, gemini_local_img])

            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            vlm_result = json.loads(clean_json)

            receiver_grids = vlm_result.get('receiver_grids', [])
            operator_grids = vlm_result.get('operator_grids', [])
            rospy.loginfo(f"💡 Gemini 決定 - 接收臂: {receiver_grids}, 操作臂: {operator_grids}")
            rospy.loginfo(f"   理由: {vlm_result.get('reasoning', '')}")

            if mode == "operator_only" and not operator_grids:
                rospy.logerr("❌ operator_only 模式但 Gemini 沒有回傳操作臂網格")
                self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_operator_grids"}))
                return

            if mode == "receiver_only" and not receiver_grids:
                rospy.logerr("❌ receiver_only 模式但 Gemini 沒有回傳接收臂網格")
                self.done_pub.publish(json.dumps({"status": "fail", "reason": "no_receiver_grids"}))
                return

            if mode == "receiver_only":
                operator_grids = []
                rospy.loginfo("   模式：receiver_only，跳過操作臂遮罩")

            if mode == "operator_only":
                receiver_grids = []
                rospy.loginfo("   模式：operator_only，跳過接收臂遮罩")

            rospy.loginfo("✂️ [4/4] SAM 精準切割...")
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

            cv2.imwrite(
                os.path.join(self.save_dir, "sam_global_mask_full.png"),
                (global_mask * 255).astype(np.uint8)
            )

            def save_final_mask(grid_ids_list, filename):
                final_mask = np.zeros_like(global_mask, dtype=bool)
                for grid_id in grid_ids_list:
                    if grid_id not in grid_dict_absolute:
                        rospy.logwarn(f"⚠️ 找不到網格代號: {grid_id}")
                        continue
                    gx1, gy1, gx2, gy2 = grid_dict_absolute[grid_id]
                    final_mask[gy1:gy2, gx1:gx2] |= global_mask[gy1:gy2, gx1:gx2]
                out_path = os.path.join(self.save_dir, filename)
                cv2.imwrite(out_path, (final_mask * 255).astype(np.uint8))
                rospy.loginfo(f"✅ 遮罩已儲存: {filename}")

            if receiver_grids:
                save_final_mask(receiver_grids, "receiver_mask.png")
            if operator_grids:
                save_final_mask(operator_grids, "operator_mask.png")

            rospy.loginfo("🎉 視覺前處理完畢！")

            self.done_pub.publish(json.dumps({
                "status": "done",
                "object_name": object_name,
                "mode": mode,
                "receiver_grids": receiver_grids,
                "operator_grids": operator_grids
            }))

        except Exception as e:
            rospy.logerr(f"管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.done_pub.publish(json.dumps({"status": "fail", "reason": str(e)}))


if __name__ == '__main__':
    node = SemanticBrainNode()
    rospy.sleep(2)
    node.run()
