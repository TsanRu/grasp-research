#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# ==========================================
# 1. 只引入 ROS 系統路徑，但不 import cv_bridge
# ==========================================
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

import rospy
import message_filters
from sensor_msgs.msg import Image
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import tf2_ros

# 引入你的 AnyGrasp
from gsnet import AnyGrasp

# ==========================================
# 2. 影像轉換函式 (無 cv_bridge)
# ==========================================
def imgmsg_to_numpy(msg):
    dtype_class = np.uint8
    channels = 1
    if "rgb8" in msg.encoding:
        dtype_class = np.uint8
        channels = 3
    elif "bgr8" in msg.encoding:
        dtype_class = np.uint8
        channels = 3
    elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
        dtype_class = np.uint16
        channels = 1
    elif "32FC1" in msg.encoding:
        dtype_class = np.float32
        channels = 1
    
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
    else:
        img = img.reshape((msg.height, msg.width))
    return img

class Config:
    def __init__(self):
        # ⚠️ 請確認模型路徑
        self.checkpoint_path = './log/checkpoint_detection.tar' 
        self.max_gripper_width = 0.075
        self.gripper_height = 0.03
        self.top_down_grasp = False
        self.debug = True  # 照你的設定開啟 debug

class AnyGraspNode:
    def __init__(self):
        rospy.init_node('anygrasp_ros_node', anonymous=True)
        
        self.cfgs = Config()
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        # 你的工作空間設定
        self.lims = [-0.2, 0.25, -0.2, 0.6, 0.4, 0.8]
        # self.lims = [-0.2, 0.35, 0.2, 1.0, 0.0, 0.8] 

        rospy.loginfo("Loading AnyGrasp Model...")
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("AnyGrasp Model Loaded!")

        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 【新增】 用來存手臂基座的真實位置
        self.arm_base_cache = {}

        self.need_detection = False
        rospy.loginfo("Ready.")

    def trigger(self):
        rospy.loginfo("Trigger received! Processing...")
        self.need_detection = True
    
    def get_arm_base_position(self, arm_name, ref_frame):
        """ 
        查詢手臂基座在 'ref_frame' (相機座標系) 下的位置 
        """
        try:
            # 嘗試查詢 TF
            base_link_name = f"{arm_name}_base_link" 
            trans = self.tf_buffer.lookup_transform(ref_frame, base_link_name, rospy.Time(0), rospy.Duration(1.0))
            
            pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            rospy.loginfo_once(f"✅ [TF成功] {arm_name} 基座座標: {pos}")
            return pos
            
        except Exception as e:
            # ⚠️ TF 失敗時的備用方案 (這是關鍵！)
            rospy.logwarn_once(f"⚠️ [TF失敗] 無法查詢 {base_link_name}，使用預設座標。錯誤: {e}")
            
            # 假設相機是 RealSense，座標系通常是: X右, Y下, Z前
            # 左手應該在相機的左邊 (X < 0)
            # 右手應該在相機的右邊 (X > 0)
            
            if "left" in arm_name:
                return np.array([-0.5, 0.2, 0.0]) # 左手預設值 (負X)
            else:
                return np.array([0.5, 0.2, 0.0])  # 右手預設值 (正X)
            
    def sort_grasps_by_reachability(self, gg, arm_name, ref_frame):
        """
        雙重過濾機制：
        1. 【角度】 Orientation: 夾爪 Z 軸必須朝向物體 (Dot Product)
        2. 【位置】 Position: 抓取點必須離手臂基座比較近 (Distance Penalty)
        """
        # 1. 取得手臂位置
        arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        
        # 2. 【最終修正版】 設定 "45度" 理想向量
        # 這樣可以同時捕捉 "側面抓取 (+X)" 和 "上方抓取 (+Y)"
        # 並且依然能完美過濾掉 "反向抓取 (-X)"
        
        if "left" in arm_name:
            # 左手：喜歡 "往右 (+X)" 或是 "往下 (+Y)"
            # 向量設為 [1.0, 1.0, 0.0]
            ideal_vector = np.array([1.0, 1.0, 0.0]) 
        else:
            # 右手：喜歡 "往左 (-X)" 或是 "往下 (+Y)"
            # 向量設為 [-1.0, 1.0, 0.0]
            ideal_vector = np.array([-1.0, 1.0, 0.0])
            
        # 歸一化
        ideal_vector = ideal_vector / np.linalg.norm(ideal_vector)
        
        rospy.loginfo(f"篩選中... 手臂: {arm_name} | 理想向量: {ideal_vector}")

        reordered_gg = []
        
        for i, grasp in enumerate(gg):
            grasp_z_axis = grasp.rotation_matrix[:, 0] 
            
            # --- 數學驗證 ---
            # 情況1 (側面抓): Grasp=[1,0,0], Ideal=[0.7, 0.7, 0] -> Score = 0.7 (高分!)
            # 情況2 (上方抓): Grasp=[0,1,0], Ideal=[0.7, 0.7, 0] -> Score = 0.7 (高分!)
            # 情況3 (反向抓): Grasp=[-1,0,0], Ideal=[0.7, 0.7, 0] -> Score = -0.7 (淘汰!)
            
            alignment_score = np.dot(grasp_z_axis, ideal_vector)
            dist = np.linalg.norm(grasp.translation - arm_base_pos)
            
            original_score = grasp.score
            w_align = 3.0   
            w_dist = 8.0
            base_score = 5.0
            
            bonus = 0.0
            penalty = 0.0
            
            # 門檻設為 0.0 即可，因為 0.7 遠大於 0.0
            if alignment_score < -0.2: 
                 grasp.score = -99.0 
                 final_score = -99.0
            else:
                 bonus = (alignment_score * w_align)
                 penalty = (dist * w_dist)
                 final_score = base_score + original_score + bonus - penalty
                 grasp.score = final_score

            # ========================================================
            # 【關鍵修改】 把計算細節存入 grasp 物件中 (以便後續印出)
            # ========================================================
            grasp.details = {
                "base": base_score,
                "orig": original_score,
                "bonus": bonus,
                "penalty": penalty,
                "align": alignment_score,
                "dist": dist,
                "w_dist": w_dist,
                "w_align": w_align
            }

            reordered_gg.append(grasp)

        reordered_gg.sort(key=lambda g: g.score, reverse=True)
        
        
        return reordered_gg

    def callback(self, color_msg, depth_msg):
        if not self.need_detection:
            return

        try:
            rospy.loginfo("Processing images...")
            
            # --- 1. 影像處理 (ROS -> Numpy) ---
            color_np = imgmsg_to_numpy(color_msg)
            if "bgr8" in color_msg.encoding:
                color_np = color_np[:, :, ::-1]
            colors = color_np.astype(np.float32) / 255.0

            depths = imgmsg_to_numpy(depth_msg)
            depths = depths.astype(np.float32)

            # --- 2. 單位與點雲轉換 (照你的邏輯) ---
            try:
                max_depth_val = np.nanmax(depths)
            except:
                max_depth_val = 0.0

            if max_depth_val > 100:
                scale = 1000.0
            else:
                scale = 1.0

            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)
            
            points_z = depths / scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            mask = (points_z > 0)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)

            rospy.loginfo(f"Point Cloud Points: {points.shape[0]}")
            
            # # ==========================================
            # # 🛡️ 新增：拯救顯卡的點雲降採樣 (Downsampling)
            # # ==========================================
            # MAX_POINTS = 30000  # AnyGrasp 最喜歡的黃金數量
            # if points.shape[0] > MAX_POINTS:
            #     rospy.loginfo(f"⚠️ 點雲太龐大！正在隨機抽樣打薄至 {MAX_POINTS} 個點...")
            #     # 隨機挑選 3 萬個不重複的索引
            #     indices = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
            #     points = points[indices]
            #     colors = colors[indices]
            # # ==========================================

            # --- 3. 執行偵測 ---
            gg, cloud = self.anygrasp.get_grasp(
                points, colors, lims=self.lims, 
                apply_object_mask=True, dense_grasp=False, collision_detection=True
            )

            if len(gg) == 0:
                rospy.logwarn("No Grasp detected!")
                self.need_detection = False
                return

            gg = gg.nms().sort_by_score()
            
            # ==========================================
            # 【關鍵修改】 呼叫方向篩選器
            # ==========================================
            target_arm = "rightarm"  # 您可以根據實際需求修改
            camera_frame_id = color_msg.header.frame_id # 通常是 "camera_color_optical_frame"
            
            # 確保有 Frame ID，防止空值
            if not camera_frame_id:
                camera_frame_id = "camera_color_optical_frame"
            
            # 執行排序，這會把那些朝向螢幕外的姿態分數扣爆
            gg = self.sort_grasps_by_reachability(gg, target_arm, camera_frame_id)
            
            rospy.loginfo(f"Grasps re-sorted for {target_arm} reachability.")
            

            # --- 4. 印出前 5 個抓取資訊 (照你的 demo 程式碼) ---
            top_k_grasps = 10
            gg_pick = gg[0:top_k_grasps]

            print("\n" + "="*40)
            print(f"Displaying Top {len(gg_pick)} Grasp Candidates:")
            print("="*40)

            for i, grasp in enumerate(gg_pick):
                print(f"--- Grasp Candidate #{i} ---")
                
                # 如果我們有存細節，就印出來
                if hasattr(grasp, 'details'):
                    d = grasp.details
                    print(f"  [分數詳情] 總分: {grasp.score:.4f}")
                    print(f"    (+) 基礎分數 : {d['base']}")
                    print(f"    (+) 原始分數 : {d['orig']:.4f} (AnyGrasp)")
                    print(f"    (+) 方向加分 : {d['bonus']:.4f} (Align={d['align']:.2f} * {d['w_align']})")
                    print(f"    (-) 距離扣分 : {d['penalty']:.4f} (Dist ={d['dist']:.2f} * {d['w_dist']})")
                else:
                    print(f"Score: {grasp.score:.4f}")
                    
                # print(f"Score: {grasp.score:.4f}")
                print(f"Gripper Width: {grasp.width:.4f} (m)")
                
                translation = grasp.translation
                print(f"Position (X, Y, Z): [{translation[0]:.8f}, {translation[1]:.8f}, {translation[2]:.8f}]")

                rotation_matrix = grasp.rotation_matrix
                # print("Orientation (Rotation Matrix):")
                # print(np.round(rotation_matrix, 3))

                try:
                    r = Rotation.from_matrix(rotation_matrix)
                    quaternion = r.as_quat() 
                    print(f"Orientation (Quaternion x,y,z,w): [{quaternion[0]:.8f}, {quaternion[1]:.8f}, {quaternion[2]:.8f}, {quaternion[3]:.8f}]")
                except Exception as e:
                    print(f"Could not convert to quaternion: {e}")
                print("-" * 20)

            # --- 5. 視覺化部分 (照你的 demo 程式碼移植) ---
            if self.cfgs.debug:
                print("Showing the BEST grasp (#0) in 3D... (Close window to continue)")
                
                # grippers = gg.to_open3d_geometry_list()
                
                grippers = [g.to_open3d_geometry() for g in gg]
                
                # # 正常的視角(視角上下顛倒)
                # trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
                # 視覺化的視角(人體直覺視角)
                trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                
                # [修正] 防止 cloud 為 None 導致程式崩潰
                if cloud is None:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points)
                    cloud.colors = o3d.utility.Vector3dVector(colors)

                cloud.transform(trans_mat)
                
                for gripper in grippers:
                    gripper.transform(trans_mat)

                # # 顯示最好的一個抓取 (grippers[0]) 和點雲
                # o3d.visualization.draw_geometries([grippers[3], cloud])
                
                # === 關鍵修改：取前 5 個 (或是更少，如果偵測到的不到 5 個) ===
                # grippers[:5] 代表取列表中的第 0 到第 4 個元素
                top_k_grippers = grippers[:top_k_grasps]

                # 將點雲與前 5 個夾爪放入同一個 list
                visualization_list = [cloud] + top_k_grippers

                # 一次畫出所有物件
                o3d.visualization.draw_geometries(visualization_list)
                
                
            #     # --- 5. 視覺化部分 ---
            # if self.cfgs.debug:
            #     print("Showing the BEST grasp (#0) in 3D... (Close window to continue)")
                
            #     # 1. 夾爪幾何 (原本的)
            #     grippers = [g.to_open3d_geometry() for g in gg]
                
            #     # 2. 點雲 (原本的)
            #     if cloud is None:
            #         cloud = o3d.geometry.PointCloud()
            #         cloud.points = o3d.utility.Vector3dVector(points)
            #         cloud.colors = o3d.utility.Vector3dVector(colors)

            #     # 3. 【關鍵新增】 為前 3 名的夾爪畫出「自身座標軸」
            #     # 這能幫助我們判斷 Z 軸 (藍色線) 到底是刺進去還是刺出來
            #     axes_list = []
            #     top_k_vis = min(len(gg), 3) # 只畫前 3 名的座標軸，避免太亂
                
            #     for i in range(top_k_vis):
            #         g = gg[i]
            #         # 在夾爪中心畫一個座標軸 (大小 0.1m)
            #         # 紅=X, 綠=Y, 藍=Z (刺出方向)
            #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                    
            #         # 將座標軸旋轉並移動到夾爪的位置
            #         axis.rotate(g.rotation_matrix, center=[0,0,0])
            #         axis.translate(g.translation)
            #         axes_list.append(axis)

            #     # 4. 視角轉換 (為了不混淆方向，我們先註解掉翻轉視角的程式碼)
            #     # 如果您習慣看翻轉後的，可以把下面這幾行解開，但要注意座標軸也會跟著轉
            #     trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
            #     cloud.transform(trans_mat)
            #     for gripper in grippers: gripper.transform(trans_mat)
            #     for axis in axes_list: axis.transform(trans_mat) # 座標軸也要轉

            #     # 5. 準備要畫的東西
            #     # 取前 5 個夾爪模型
            #     top_k_grippers = grippers[:top_k_grasps]
                
            #     # 全部加在一起：點雲 + 夾爪 + 座標軸
            #     visualization_list = [cloud] + top_k_grippers + axes_list

            #     # 6. 畫出來
            #     o3d.visualization.draw_geometries(visualization_list)

            self.need_detection = False

        except Exception as e:
            rospy.logerr(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.need_detection = False

if __name__ == '__main__':
    node = AnyGraspNode()
    
    # 自動觸發一次
    rospy.sleep(2)
    node.trigger()
    
    rospy.spin()