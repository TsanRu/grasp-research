# ai_planner.py
import os
import sys
import warnings

# --- 強制過濾雜訊 (Muzzle) ---
# 1. 關閉 TensorFlow/GRPC 的底層日誌
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# 2. 攔截 Python 的警告訊息 (FutureWarning 等)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # 這一行對 subprocess 特別有效
# ---------------------------

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import PIL.Image

# 載入 API Key
# load_dotenv()
# genai.configure(api_key=os.getenv("AIzaSyC1KGWfg7OoXHVyacPpzObWWQJg5NMDpHI"))

MY_TEST_KEY = "AIzaSyBrJiv3YmFDBG5eWdBrUfkxOOUq7VUtMRU" 
genai.configure(api_key=MY_TEST_KEY)

# --- 這裡定義 AI 的 "世界觀" ---
# 我們告訴它有哪些 "變數積木" 可以用，不能自己發明
ROBOT_SYSTEM_PROMPT = """
你是一個雙臂機器人任務規劃師。你的任務是根據「使用者指令」與「圖片場景」，生成對應的 JSON 任務列表。

【可用變數清單 (Strict String Tokens)】
你只能使用以下字串作為參數：
- 抓取相關: "grasp_approach_pose", "grasp_pose"
- 交接相關: "handover_approach_pose_l", "handover_approach_pose_r", "handover_pose_l", "handover_pose_r"
- 撤退相關: "retreat_pose_l", "retreat_pose_r"

【輸出格式規範】
請嚴格遵守 JSON 結構，並 **嚴格按照以下順序排列 Key**：
1. 移動: {"desc": "...", "type": "dual_p2p", "left": "...", "right": "..."}
2. 單臂移動: {"desc": "...", "type": "right_cartesian" (或 left), "waypoints": ["..."]}
3. 動作: {"desc": "...", "type": "action", "action": "grasp" (或 release), "arm": "left" (或 right)}

【SOP 邏輯 (運動策略核心)】
請嚴格遵守以下運動學原則：
1. **短距離/精細操作 (靠近/下探/抬起)**：必須使用 `_cartesian` (直線移動)。
2. **長距離/大範圍移動 (前往交接點/撤退)**：必須使用 `_p2p` (關節移動)。

【核心邏輯規則 (SOP)】
請分析指令判斷方向，並嚴格照抄以下設定 (包含 type 類型)：

情況 A：如果指令是「右手傳給左手」 (Right to Left)
1. [雙臂移動] 雙臂就位: type="dual_p2p", left="handover_approach_pose_l", right="grasp_approach_pose"
2. [單臂移動] 右臂直線下探: type="right_cartesian", waypoints=["grasp_pose"]
3. [動作] 右臂抓取: type="action", action="grasp", arm="right"
4. [單臂移動] 右臂直線抬起: type="right_cartesian", waypoints=["grasp_approach_pose"]
5. [單臂移動] 右臂前往交接: type="right_p2p", waypoints=["handover_pose_r"]
6. [單臂移動] 左臂直線靠近: type="left_cartesian", waypoints=["handover_pose_l"]
7. [動作] 左臂抓取: type="action", action="grasp", arm="left"
8. [動作] 右臂釋放: type="action", action="release", arm="right"
9. [雙臂移動] 雙臂撤退: type="dual_p2p", left="retreat_pose_l", right="retreat_pose_r"

情況 B：如果指令是「左手傳給右手」 (Left to Right)
1. [雙臂移動] 雙臂就位: type="dual_p2p", left="grasp_approach_pose", right="handover_approach_pose_r"
2. [單臂移動] 左臂直線下探: type="left_cartesian", waypoints=["grasp_pose"]
3. [動作] 左臂抓取: type="action", action="grasp", arm="left"
4. [單臂移動] 左臂直線抬起: type="left_cartesian", waypoints=["grasp_approach_pose"]
5. [單臂移動] 左臂前往交接: type="left_p2p", waypoints=["handover_pose_l"]
6. [單臂移動] 右臂直線靠近: type="right_cartesian", waypoints=["handover_pose_r"]
7. [動作] 右臂抓取: type="action", action="grasp", arm="right"
8. [動作] 左臂釋放: type="action", action="release", arm="left"
9. [雙臂移動] 雙臂撤退: type="dual_p2p", left="retreat_pose_l", right="retreat_pose_r"

注意：請直接回傳 JSON Array，不要 Markdown。
"""

class AIPlanner:
    def __init__(self, model_name='gemini-flash-latest'):
        self.model = genai.GenerativeModel(model_name)
    
    def get_script(self, image_path, instruction):
        try:
            img = PIL.Image.open(image_path)
            
            # 組合 Prompt: 系統規則 + 使用者指令
            prompt = f"{ROBOT_SYSTEM_PROMPT}\n使用者指令: {instruction}"
            
            # 呼叫 Gemini
            response = self.model.generate_content([prompt, img])
            
            # 清理字串 (去掉可能出現的 ```json ... ```)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            
            return json.loads(clean_json)
        except Exception as e:
            print(f"AI 生成失敗: {e}")
            return None

# 測試用
if __name__ == "__main__":
    # 檢查是否由主程式呼叫 (是否有傳入參數)
    if len(sys.argv) >= 3:
        # 從命令行參數讀取 (這才是跟 main_robot.py 對接的關鍵)
        image_path = sys.argv[1]
        instruction = sys.argv[2]
    else:
        # 沒參數時的預設測試 (方便您單獨跑這支程式 debug)
        image_path = "test_scene.png"
        instruction = "將鎚子從右手交給左手"
    
    planner = AIPlanner()
    script = planner.get_script(image_path, instruction)
    
    if script:
        # 【修正】使用 json.dumps 確保輸出雙引號的標準 JSON 字串
        # ensure_ascii=False 讓中文字正常顯示
        print(json.dumps(script, ensure_ascii=False, indent=4))
    else:
        sys.exit(1)
