import os
import cv2
import torch
import numpy as np
from PIL import Image
import warnings
from transformers import pipeline, SamModel, SamProcessor

warnings.filterwarnings('ignore')

# 1. 匯入你的 VLM 大腦
from semantic_grasp_analyzer import SemanticGraspAnalyzer

class HierarchicalLangSAM:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📦 正在載入 OWL-v2 偵測器與 SAM 分割模型 ({self.device})...")
        # 使用 OWL-v2 作為偵測引擎
        self.detector = pipeline(
            model="google/owlv2-base-patch16-ensemble", 
            task="zero-shot-object-detection", 
            device=self.device
        )
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    def predict(self, image_pil, parent_prompt, part_prompt):
        """
        層次化分割核心邏輯：
        1. 先在全圖找母體 (Parent, 例如 'hammer')。
        2. 根據母體 BBox 裁切圖片，物理隔離背景干擾。
        3. 在裁切圖中找局部 (Part, 例如 'hammer head')。
        4. SAM 進行精準分割。
        5. 座標還原回原圖。
        """
        # ==========================================
        # 步驟 1：在全圖尋找母體 (Parent)
        # ==========================================
        parent_preds = self.detector(image_pil, candidate_labels=[parent_prompt])
        if not parent_preds:
            print(f"   ⚠️ 找不到母體物件: '{parent_prompt}'，分割失敗。")
            return [], [], [], []
        
        # 取分數最高的母體
        best_parent = max(parent_preds, key=lambda x: x['score'])
        p_box = best_parent['box']
        
        # 準備裁切，加上Padding緩衝區，避免切太死
        w, h = image_pil.size
        pad = 20 
        left = max(0, int(p_box['xmin']) - pad)
        top = max(0, int(p_box['ymin']) - pad)
        right = min(w, int(p_box['xmax']) + pad)
        bottom = min(h, int(p_box['ymax']) + pad)

        # ==========================================
        # 步驟 2：物理隔離 (Crop)
        # ==========================================
        # 這是最關鍵的一步，把右上角機器手臂那些雜訊全部剪掉
        crop_img = image_pil.crop((left, top, right, bottom))
        print(f"   ✂️ 成功鎖定母體 '{parent_prompt}'，已執行裁切聚焦 (Offset: {left}, {top})")

        # ==========================================
        # 步驟 3：在乾淨的裁切圖中尋找局部 (Part)
        # ==========================================
        part_preds = self.detector(crop_img, candidate_labels=[part_prompt])
        if not part_preds:
            print(f"   ⚠️ 在 '{parent_prompt}' 內部找不到部位: '{part_prompt}'")
            return [], [], [], []
        
        # 取裁切圖中分數最高的部位
        best_part = max(part_preds, key=lambda x: x['score'])
        p_box_crop = best_part['box']
        # 這是相對於「小圖」的座標
        part_bbox_crop = [p_box_crop['xmin'], p_box_crop['ymin'], p_box_crop['xmax'], p_box_crop['ymax']]

        # ==========================================
        # 步驟 4：SAM 分割 (針對小圖)
        # ==========================================
        # SAM 在乾淨環境下，分割會非常精準
        inputs = self.sam_processor(crop_img, input_boxes=[[part_bbox_crop]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs.original_sizes.cpu(), 
            inputs.reshaped_input_sizes.cpu()
        )
        # 拿到裁切圖的 Boolean Mask
        crop_mask_np = masks[0][0][0].numpy() 

        # ==========================================
        # 步驟 5：座標還原 (Project Back)
        # ==========================================
        full_mask = np.zeros((h, w), dtype=bool)
        # 利用偏移量，將小 Mask 貼回原圖切片位置
        full_mask[top:bottom, left:right] = crop_mask_np

        # 還原絕對座標 (後續抓取可能需要)
        abs_bbox = [part_bbox_crop[0]+left, part_bbox_crop[1]+top, part_bbox_crop[2]+left, part_bbox_crop[3]+top]
        
        # 格式對齊原本的 LangSAM 輸出 (Tensor List)
        return [torch.from_numpy(full_mask).to(self.device)], [abs_bbox], [part_prompt], [best_part['score']]

def main():
    IMAGE_PATH = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data/color.png"
    SAVE_DIR = os.path.dirname(IMAGE_PATH)

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到圖片: {IMAGE_PATH}")
        return

    # ==========================================
    # 階段 1：啟動大腦 (從 VLM 獲取通用策略)
    # ==========================================
    print("🧠 [階段 1] 呼叫 VLM 思考抓取策略...")
    analyzer = SemanticGraspAnalyzer()
    
    # 這是你的自定義指令
    custom_instruction = "桌上有一把鎚子，我想要右手先拿起來交給左手，請幫我分配抓取部位。"
    vlm_result = analyzer.analyze_image(IMAGE_PATH, instruction=custom_instruction)
    
    if not vlm_result: return
    
    # 🌟 核心修改 1：動態提取大腦辨識出的英文物件名稱 (作為Parent) 🌟
    # 如果 VLM 沒輸出 object_name，預設用 'object' 防止崩潰
    parent_obj = vlm_result.get('object_name', 'object')
        
    print(f"✅ VLM 辨識母體為: '{parent_obj}'")
    print(f"✅ VLM 決定左手抓局部: '{vlm_result['left_part']}'")
    print(f"✅ VLM 決定右手抓局部: '{vlm_result['right_part']}'")
    print(f"💡 理由: {vlm_result['reasoning']}\n")

    # ==========================================
    # 階段 2：啟動層次化視覺皮層 (改用新類別)
    # ==========================================
    print("👁️ [階段 2] 載入層次化視覺模型 (OWL-v2 + SAM)...")
    # 類別名稱我改得更有語義：HierarchicalLangSAM
    model = HierarchicalLangSAM() 
    image_pil = Image.open(IMAGE_PATH).convert("RGB")

    # 🌟 核心修改 2：修改generate函式，接收 parent 和 part 兩個提示詞 🌟
    def generate_and_save_mask(parent_prompt, part_prompt, output_filename):
        print(f"\n🔍 正在分割: [母體]'{parent_prompt}' -> [局部]'{part_prompt}' ...")
        # 呼叫層次化 predict
        masks, boxes, phrases, logits = model.predict(image_pil, parent_prompt, part_prompt)
        
        if len(masks) == 0:
            print(f"⚠️ 無法生成 '{part_prompt}' 的遮罩。\n")
            return
            
        mask_np = masks[0].cpu().numpy() # 拿第一個 mask
        mask_img = (mask_np * 255).astype(np.uint8)
        cv2.imwrite(output_filename, mask_img)
        print(f"🎉 成功儲存精準遮罩: {output_filename}")

    # ==========================================
    # 階段 3：執行全自動通用切割
    # ==========================================
    # 這裡我們完全不需要硬編碼 "hammer"，全部動態從 VLM 結果獲取！
    
    left_mask_path = os.path.join(SAVE_DIR, "left_mask.png")
    generate_and_save_mask(parent_obj, vlm_result["left_part"], left_mask_path)

    right_mask_path = os.path.join(SAVE_DIR, "right_mask.png")
    generate_and_save_mask(parent_obj, vlm_result["right_part"], right_mask_path)
    
    print("\n🚀 視覺 Pipeline 執行完畢！所有 Mask 已就緒。")

if __name__ == "__main__":
    main()