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
        self.max_gripper_width = 0.085
        self.gripper_height = 0.045
        self.top_down_grasp = False
        self.debug = True  # 照你的設定開啟 debug

class AnyGraspNode:
    def __init__(self):
        rospy.init_node('anygrasp_ros_node', anonymous=True)
        
        self.cfgs = Config()
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        
        # 您的工作空間設定 (可以在這裡隨時微調)
        self.lims = [-0.2, 0.25, -0.2, 0.6, 0.4, 0.8]

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
        
        self.arm_base_cache = {}
        self.need_detection = False
        rospy.loginfo("Ready.")

    def trigger(self):
        rospy.loginfo("Trigger received! Processing...")
        self.need_detection = True
    
    def get_arm_base_position(self, arm_name, ref_frame):
        try:
            base_link_name = f"{arm_name}_base_link" 
            trans = self.tf_buffer.lookup_transform(ref_frame, base_link_name, rospy.Time(0), rospy.Duration(1.0))
            pos = np.array([
                trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
            ])
            rospy.loginfo_once(f"✅ [TF成功] {arm_name} 基座座標: {pos}")
            return pos
        except Exception as e:
            rospy.logwarn_once(f"⚠️ [TF失敗] 無法查詢 {base_link_name}，使用預設座標。錯誤: {e}")
            if "left" in arm_name: return np.array([-0.5, 0.2, 0.0])
            else: return np.array([0.5, 0.2, 0.0])
            
    def sort_grasps_by_reachability(self, gg, arm_name, ref_frame):
        # 雖然暫時不用，但保留函式以便之後加回來
        arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        if "left" in arm_name:
            ideal_vector = np.array([1.0, 1.0, 0.0]) 
        else:
            ideal_vector = np.array([-1.0, 1.0, 0.0])
            
        ideal_vector = ideal_vector / np.linalg.norm(ideal_vector)
        reordered_gg = []
        
        for i, grasp in enumerate(gg):
            grasp_z_axis = grasp.rotation_matrix[:, 0] 
            alignment_score = np.dot(grasp_z_axis, ideal_vector)
            dist = np.linalg.norm(grasp.translation - arm_base_pos)
            
            original_score = grasp.score
            w_align = 3.0; w_dist = 8.0; base_score = 5.0
            bonus = 0.0; penalty = 0.0
            
            if alignment_score < -0.2: 
                 grasp.score = -99.0; final_score = -99.0
            else:
                 bonus = (alignment_score * w_align)
                 penalty = (dist * w_dist)
                 final_score = base_score + original_score + bonus - penalty
                 grasp.score = final_score

            grasp.details = {
                "base": base_score, "orig": original_score, "bonus": bonus,
                "penalty": penalty, "align": alignment_score, "dist": dist,
                "w_dist": w_dist, "w_align": w_align
            }
            reordered_gg.append(grasp)

        reordered_gg.sort(key=lambda g: g.score, reverse=True)
        return reordered_gg

    def callback(self, color_msg, depth_msg):
        if not self.need_detection:
            return

        try:
            rospy.loginfo("Processing images...")
            self.need_detection = False
            
            # --- 1. 影像處理 ---
            color_np = imgmsg_to_numpy(color_msg)
            if "bgr8" in color_msg.encoding: color_np = color_np[:, :, ::-1]
            colors = color_np.astype(np.float32) / 255.0

            depths = imgmsg_to_numpy(depth_msg).astype(np.float32)

            try: max_depth_val = np.nanmax(depths)
            except: max_depth_val = 0.0
            scale = 1000.0 if max_depth_val > 100 else 1.0

            xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
            points_z = depths / scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            mask = (points_z > 0)
            points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
            colors = colors[mask].astype(np.float32)

            rospy.loginfo(f"Point Cloud Points: {points.shape[0]}")
            
            # ==========================================
            # 🛡️ 新增：拯救顯卡的點雲降採樣 (Downsampling)
            # ==========================================
            MAX_POINTS = 30000  # AnyGrasp 最喜歡的黃金數量
            if points.shape[0] > MAX_POINTS:
                rospy.loginfo(f"⚠️ 點雲太龐大！正在隨機抽樣打薄至 {MAX_POINTS} 個點...")
                # 隨機挑選 3 萬個不重複的索引
                indices = np.random.choice(points.shape[0], MAX_POINTS, replace=False)
                points = points[indices]
                colors = colors[indices]
            # ==========================================

            # --- 3. 執行偵測 ---
            gg, cloud = self.anygrasp.get_grasp(
                points, colors, lims=self.lims, 
                apply_object_mask=True, dense_grasp=True, collision_detection=True
            )

            # 【防呆修改】 加入 gg is None 防止空值崩潰
            if gg is None or len(gg) == 0:
                rospy.logwarn("目前範圍內沒有偵測到可抓取的目標！請檢查 lims 設定。")
                self.need_detection = False
                return

            # NMS 篩選並依據模型預測的原始分數排序
            gg = gg.nms().sort_by_score()
            
            # ==========================================
            # 【暫時關閉過濾邏輯】
            # ==========================================
            # target_arm = "rightarm"
            # camera_frame_id = color_msg.header.frame_id if color_msg.header.frame_id else "camera_color_optical_frame"
            # gg = self.sort_grasps_by_reachability(gg, target_arm, camera_frame_id)
            # rospy.loginfo(f"Grasps re-sorted for {target_arm} reachability.")
            rospy.loginfo("目前顯示的是 AnyGrasp 原始生成姿態 (未經過濾)。")

            # --- 4. 印出前 10 個抓取資訊 ---
            top_k_grasps = 20
            gg_pick = gg[0:top_k_grasps]

            print("\n" + "="*40)
            print(f"Displaying Top {len(gg_pick)} Grasp Candidates:")
            print("="*40)

            for i, grasp in enumerate(gg_pick):
                print(f"--- Grasp Candidate #{i} ---")
                if hasattr(grasp, 'details'):
                    d = grasp.details
                    print(f"  [分數詳情] 總分: {grasp.score:.4f}")
                    # ... 省略詳細印出，因為目前不過濾所以不會有 details
                else:
                    print(f"Score (Original): {grasp.score:.4f}")
                    
                print(f"Gripper Width: {grasp.width:.4f} (m)")
                translation = grasp.translation
                print(f"Position (X, Y, Z): [{translation[0]:.8f}, {translation[1]:.8f}, {translation[2]:.8f}]")

                try:
                    r = Rotation.from_matrix(grasp.rotation_matrix)
                    quaternion = r.as_quat() 
                    print(f"Orientation (Quaternion): [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
                except Exception as e:
                    pass
                print("-" * 20)

           # --- 5. 視覺化部分 (照你的 demo 程式碼移植) ---
            if self.cfgs.debug:
                print("Showing the grasp candidates in 3D... (Close window to continue)")
                
                grippers = [g.to_open3d_geometry() for g in gg]
                
                if cloud is None:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points)
                    cloud.colors = o3d.utility.Vector3dVector(colors)

                # ==========================================
                # 【新增】 建立 Bounding Box 並裁切點雲
                # ==========================================
                min_bound = np.array([self.lims[0], self.lims[2], self.lims[4]])
                max_bound = np.array([self.lims[1], self.lims[3], self.lims[5]])
                
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                
                # 1. 直接用原本的框把框外的點雲裁掉
                cloud = cloud.crop(bbox)

                # 2. ⚠️【防呆修復】將 AxisAligned 的框轉換成純線條 (LineSet)，這樣就可以任意旋轉不會報錯了
                bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
                bbox_lines.paint_uniform_color([1.0, 0.0, 0.0]) # 塗成紅色
                # ==========================================

                # 視覺化的視角(人體直覺視角)
                trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                
                # 翻轉點雲、純線條紅框、與夾爪
                cloud.transform(trans_mat)
                bbox_lines.transform(trans_mat) # 這裡就不會報錯了！
                
                for gripper in grippers:
                    gripper.transform(trans_mat)

                top_k_grippers = grippers[:top_k_grasps]

                # 將乾淨的點雲、紅色純線條框、前 10 個夾爪放入同一個 list 顯示
                visualization_list = [cloud, bbox_lines] + top_k_grippers

                o3d.visualization.draw_geometries(visualization_list)

        except Exception as e:
            rospy.logerr(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.need_detection = False

if __name__ == '__main__':
    node = AnyGraspNode()
    rospy.sleep(2)
    node.trigger()
    rospy.spin()