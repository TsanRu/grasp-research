#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2 

# ==========================================
# 🛡️ 1. 安全引入 ROS (防止污染 AnyGrasp 環境)
# ==========================================
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    import message_filters
    from sensor_msgs.msg import Image
    import tf2_ros
    print("✅ 成功跨界連接 ROS Noetic！")
except ImportError:
    print("❌ 找不到 ROS，請確認終端機有 source /opt/ros/noetic/setup.bash")

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# 匯入你的 AnyGrasp 核心
from gsnet import AnyGrasp

# ==========================================
# 📸 2. 影像轉換函式 (繞過 cv_bridge 衝突)
# ==========================================
def imgmsg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 1
    if "rgb8" in msg.encoding or "bgr8" in msg.encoding:
        channels = 3
    elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
        dtype_class = np.uint16
    elif "32FC1" in msg.encoding:
        dtype_class = np.float32
    
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
    else:
        img = img.reshape((msg.height, msg.width))
    return img

# ==========================================
# ⚙️ 3. AnyGrasp 參數設定
# ==========================================
class Config:
    def __init__(self):
        self.checkpoint_path = './log/checkpoint_detection.tar' 
        self.max_gripper_width = 0.085
        self.gripper_height = 0.035
        self.top_down_grasp = False # 允許各種角度的抓取
        self.debug = True           # 開啟 Open3D 視覺化

# ==========================================
# 🦾 4. 主節點類別
# ==========================================
class AnyGraspMaskedNode:
    def __init__(self):
        rospy.init_node('anygrasp_masked_node', anonymous=True)
        
        self.cfgs = Config()
        # ⚠️ 請確保這裡是你真實相機 (或 Gazebo 相機) 的內參！
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        
        # Node 1 (大腦) 存檔遮罩的路徑
        self.mask_dir = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"
        self.target_arm = "left" 

        rospy.loginfo("🦾 正在載入 AnyGrasp 物理引擎...")
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("✅ AnyGrasp 載入完成！")

        # 裝上天線，監聽相機頻道
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        # 時間同步器 (確保彩色與深度是同一個瞬間)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.callback)
        
        self.need_detection = False
        rospy.loginfo("🤖 小腦節點就緒，等待大腦 trigger...")

    def trigger(self, arm="left"):
        """呼叫這個函式來啟動實體抓取計算"""
        self.target_arm = arm
        rospy.loginfo(f"⚡ 收到 Trigger！準備為 {arm} 手臂計算抓取姿態...")
        self.need_detection = True

    def callback(self, color_msg, depth_msg):
        if not self.need_detection:
            return

        try:
            rospy.loginfo("⚙️ 啟動 3D 幾何管線與防撞計算...")
            self.need_detection = False
            
            # --- A. 處理相機即時影像 ---
            color_np = imgmsg_to_numpy(color_msg)
            if "bgr8" in color_msg.encoding: color_np = color_np[:, :, ::-1]
            colors = color_np.astype(np.float32) / 255.0

            depths = imgmsg_to_numpy(depth_msg).astype(np.float32)
            max_depth_val = np.nanmax(depths) if np.nanmax(depths) > 0 else 0.0
            scale = 1000.0 if max_depth_val > 100 else 1.0

            # --- B. 讀取大腦做好的 2D 遮罩 ---
            mask_filename = "left_mask.png" if self.target_arm == "left" else "right_mask.png"
            mask_path = os.path.join(self.mask_dir, mask_filename)
            
            if not os.path.exists(mask_path):
                rospy.logerr(f"❌ 找不到遮罩檔案: {mask_path}。請確認 Node 1 (大腦) 已經執行完畢！")
                return

            semantic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 確保遮罩大小跟現在的相機畫面一樣大
            if semantic_mask.shape != (color_np.shape[0], color_np.shape[1]):
                semantic_mask = cv2.resize(semantic_mask, (color_np.shape[1], color_np.shape[0]))
            
            boolean_mask = semantic_mask > 127

            # --- C. 降維打擊升維：2D $\to$ 3D 投影 ---
            xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
            points_z = depths / scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            valid_depth_mask = (points_z > 0)
            
            # 🌟 核心過濾：只保留大腦指定範圍內的有效點雲 🌟
            final_mask = valid_depth_mask & boolean_mask

            # 生成【全域點雲】(包含桌子，給 AnyGrasp 防撞用)
            points_full = np.stack([points_x, points_y, points_z], axis=-1)[valid_depth_mask].astype(np.float32)
            colors_full = colors[valid_depth_mask].astype(np.float32)

            # 生成【局部目標點雲】(只有大腦指定的部位，用來算結界)
            points_target = np.stack([points_x, points_y, points_z], axis=-1)[final_mask].astype(np.float32)

            if points_target.shape[0] < 50:
                rospy.logwarn("⚠️ 大腦指定的區域內缺乏有效的深度點雲 (可能反光或太遠)！")
                return

            # --- D. 建立 3D 物理結界 (Dynamic Lims) ---
            min_bound = points_target.min(axis=0)
            max_bound = points_target.max(axis=0)

            # 加上 3 公分的 Padding，讓夾爪有足夠的空間靠近物體
            pad = 0 
            dynamic_lims = [
                min_bound[0] - pad, max_bound[0] + pad,
                min_bound[1] - pad, max_bound[1] + pad,
                min_bound[2] - pad, max_bound[2] + pad
            ]
            rospy.loginfo(f"📏 動態 3D 結界生成完畢: {dynamic_lims}")

            # --- E. 呼叫 AnyGrasp 進行物理計算 ---
            rospy.loginfo("🧠 AnyGrasp 正在結界內尋找最佳姿態...")
            gg, cloud = self.anygrasp.get_grasp(
                points_full, 
                colors_full, 
                lims=dynamic_lims,       # 🌟 鎖定在 3D 結界內
                apply_object_mask=False, # 停用預設過濾，改用我們的精準結界
                dense_grasp=True,        # 密集生成，確保找到好角度
                collision_detection=True # 開啟防撞 (感知 points_full 裡的桌子)
            )

            if gg is None or len(gg) == 0:
                rospy.logwarn(f"⚠️ AnyGrasp 在指定的結界內找不到安全的抓取姿態！")
                return

            # NMS 篩選並依據分數排序
            gg = gg.nms().sort_by_score()
            best_grasp = gg[0]
            
            # --- F. 輸出結果 ---
            print("\n" + "="*50)
            print(f"🎯 {self.target_arm.upper()} 手臂 (Top-1) 目標抓取位姿:")
            print("="*50)
            print(f"  ⭐ 分數: {best_grasp.score:.4f}")
            print(f"  📏 夾爪寬度: {best_grasp.width:.4f} m")
            print(f"  📍 位置 (X, Y, Z): [{best_grasp.translation[0]:.4f}, {best_grasp.translation[1]:.4f}, {best_grasp.translation[2]:.4f}]")
            
            try:
                r = Rotation.from_matrix(best_grasp.rotation_matrix)
                q = r.as_quat() 
                print(f"  🔄 旋轉 (四元數): [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
            except: pass
            print("="*50 + "\n")

            # --- G. 3D 視覺化 (除錯用) ---
            if self.cfgs.debug:
                print(f"👀 開啟 3D 視覺化：紅框為 3D 結界，夾爪必須在紅框內，綠色為原始遮罩點雲。")
                top_k = min(5, len(gg))
                grippers = [g.to_open3d_geometry() for g in gg[:top_k]]
                
                if cloud is None:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points_full)
                    cloud.colors = o3d.utility.Vector3dVector(colors_full)

                # 畫出紅色結界 Bounding Box
                min_b = np.array([dynamic_lims[0], dynamic_lims[2], dynamic_lims[4]])
                max_b = np.array([dynamic_lims[1], dynamic_lims[3], dynamic_lims[5]])
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
                
                # 裁切點雲以方便肉眼觀察
                cloud = cloud.crop(bbox)

                # ==========================================
                # 🌟 [新增] 建立「最原始遮罩」點雲，並塗成鮮豔的綠色！🌟
                # ==========================================
                mask_cloud = o3d.geometry.PointCloud()
                mask_cloud.points = o3d.utility.Vector3dVector(points_target)
                mask_cloud.paint_uniform_color([0.0, 1.0, 0.0]) # 亮綠色
                # ==========================================

                bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                bbox_lines.paint_uniform_color([1.0, 0.0, 0.0]) # 紅色線條

                # Open3D 顯示翻轉修正
                trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                cloud.transform(trans_mat)
                mask_cloud.transform(trans_mat) # 綠色點雲也要翻轉
                bbox_lines.transform(trans_mat)
                for gripper in grippers:
                    gripper.transform(trans_mat)

                # 把 mask_cloud 放進顯示陣列裡
                o3d.visualization.draw_geometries([cloud, mask_cloud, bbox_lines] + grippers)

        except Exception as e:
            rospy.logerr(f"小腦管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    node = AnyGraspMaskedNode()
    rospy.sleep(2) # 等待 ROS 連線穩定
    
    # 測試觸發：計算左手的抓取姿態
    node.trigger(arm="right")
    
    rospy.spin()