#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# ==========================================
# 1. 引入 ROS 系統路徑
# ==========================================
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

import rospy
import message_filters
from sensor_msgs.msg import Image
import tf2_ros
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# 引入你的 AnyGrasp
from gsnet import AnyGrasp
# 引入 ROS 訊息類型
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

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
        self.gripper_height = 0.035
        self.top_down_grasp = False
        self.debug = True  # 開啟 debug 以顯示 Open3D 視窗

class AnyGraspNode:
    def __init__(self):
        rospy.init_node('anygrasp_ros_node', anonymous=True)
        
        self.cfgs = Config()
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        # 你的工作空間設定 (這些一直都在，不會不見)
        # self.lims = [-0.2, 0.25, 0.0, 0.6, 0.4, 0.8] 
        self.lims = [-0.2, 0.25, -0.2, 0.6, 0.4, 0.8]

        rospy.loginfo("Loading AnyGrasp Model...")
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("AnyGrasp Model Loaded!")

        # --- 訂閱相機 (眼睛) ---
        # 這些必須留著，不然你無法取得影像數據
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 新增：訂閱觸發指令 (耳朵) ---
        # rospy.Subscriber("/system/trigger_detection", Bool, self.trigger_callback)
        rospy.Subscriber("/system/trigger_detection", String, self.trigger_callback)
        
        #  新增發布 JSON 計畫書的 Publisher
        self.plan_pub = rospy.Publisher("/anygrasp/handover_plan", String, queue_size=1)

        # --- 新增：發布結果 (嘴巴) ---
        # self.grasp_pub = rospy.Publisher("/anygrasp/target_pose", PoseStamped, queue_size=1)
        self.grasp_pub = rospy.Publisher("/anygrasp/target_poses", PoseArray, queue_size=1)
        
        self.need_detection = False
        self.target_arm_request = "rightarm" # 預設值，避免空值錯誤
        rospy.loginfo("AnyGrasp Ready. Waiting for command from /system/trigger_detection...")

    def trigger_callback(self, msg):
        """ 
        當收到 String 指令時：
        1. 儲存指令內容 (leftarm/rightarm)
        2. 開啟偵測旗標 
        """
        rospy.loginfo(f"Command Received: Detect for '{msg.data}'")
        self.target_arm_request = msg.data # 存下來！
        self.need_detection = True
        
    # ==========================================
    #  新增物件分類器
    # ==========================================
    def categorize_object(self, obj_name):
        obj_name = obj_name.lower()
        if obj_name in ["hammer", "scissors", "banana"]:
            return "LONG"
        elif obj_name in ["bowl", "mug"]:
            return "RIM"
        else:
            return "SYMMETRIC" # cracker_box, tomato_soup_can 預設為對稱類
        
    #  加入輔助函式：將 AnyGrasp 的 Grasp 物件轉為標準 ROS Pose
    def to_ros_pose(self, grasp):
        p = Pose()
        p.position.x, p.position.y, p.position.z = grasp.translation
        q = Rotation.from_matrix(grasp.rotation_matrix).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    #  加入輔助函式：將 ROS Pose 轉為 Python Dict (為了 JSON)
    def pose_to_dict(self, pose):
        return {
            'position': {'x': pose.position.x, 'y': pose.position.y, 'z': pose.position.z},
            'orientation': {'x': pose.orientation.x, 'y': pose.orientation.y, 'z': pose.orientation.z, 'w': pose.orientation.w}
        }

    # #  將雙臂配對邏輯搬入視覺端，並生成全知計畫書
    # def generate_handover_plan(self, gg):
    #     num_poses = len(gg)
    #     if num_poses < 2: return []
            
    #     rospy.loginfo(f"🔍 [視覺端大腦] 開始評估並建立雙臂交接計畫書...")
    #     grouped_pairs = {} 

    #     for i in range(num_poses):
    #         for j in range(num_poses):
    #             if i == j: continue 
                
    #             grasp_R = gg[i]
    #             grasp_L = gg[j]
                
    #             pos_R = grasp_R.translation
    #             pos_L = grasp_L.translation
                
    #             dist = np.linalg.norm(pos_R - pos_L)
    #             if dist < 0.15: continue
                    
    #             vec_R = grasp_R.rotation_matrix[:, 2] 
    #             vec_L = grasp_L.rotation_matrix[:, 2]
    #             dot_approach = np.dot(vec_R, vec_L)
    #             if dot_approach > 0.3: continue
                    
    #             score_dist = min(dist, 0.15) * 30.0 
    #             score_angle = (1.0 - dot_approach) * 5.0 
    #             score_height = pos_R[2] * 10.0 
    #             total_score = score_dist + score_angle + score_height + (grasp_R.score + grasp_L.score) * 0.5
                
    #             pose_R = self.to_ros_pose(grasp_R)
    #             pose_L = self.to_ros_pose(grasp_L)
                
    #             if i not in grouped_pairs:
    #                 grouped_pairs[i] = {
    #                     'orig_idx_R': i,
    #                     'pose_R': pose_R,
    #                     'max_score': -float('inf'),
    #                     'left_candidates': []       
    #                 }
                
    #             grouped_pairs[i]['left_candidates'].append({
    #                 'orig_idx_L': j,
    #                 'score': total_score,
    #                 'pose_L': pose_L
    #             })
                
    #             if total_score > grouped_pairs[i]['max_score']:
    #                 grouped_pairs[i]['max_score'] = total_score
                    
    #     if not grouped_pairs: return []
            
    #     for i in grouped_pairs:
    #         grouped_pairs[i]['left_candidates'].sort(key=lambda x: x['score'], reverse=True)
            
    #     ranked_groups = list(grouped_pairs.values())
    #     ranked_groups.sort(key=lambda x: x['max_score'], reverse=True)

    #     # ==========================================
    #     # 💡 [關鍵修改處] 拔除空中計算，只打包純淨的相機視角清單
    #     # ==========================================
    #     json_plan = []
    #     for group in ranked_groups:
    #         pose_R_table = group['pose_R']
    #         plan_item = {
    #             'orig_idx_R': group['orig_idx_R'],
    #             'max_score': group['max_score'],
    #             # 這裡只傳遞相機視角下的右手目標
    #             'pose_R_table': self.pose_to_dict(pose_R_table), 
    #             'left_candidates': []
    #         }
            
    #         for cand in group['left_candidates']:
    #             pose_L_table = cand['pose_L']
    #             # 這裡只傳遞相機視角下的左手備案目標
    #             plan_item['left_candidates'].append({
    #                 'orig_idx_L': cand['orig_idx_L'],
    #                 'score': cand['score'],
    #                 'pose_L_table': self.pose_to_dict(pose_L_table)
    #             })
    #         json_plan.append(plan_item)
            
    #     return json_plan
    
    def get_arm_specific_grasps(self, gg, arm_name, ref_frame):
        """
        針對特定手臂，篩選出符合該手臂人體工學（順手方向）的姿態，
        並保留它在全局 gg 中的原始編號 (以便對應 Open3D 顏色)。
        """
        arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        
        # 定義手臂的理想延伸方向
        if "left" in arm_name:
            ideal_vector = np.array([1.0, 1.0, 0.0]) 
        else:
            ideal_vector = np.array([-1.0, 1.0, 0.0])
            
        ideal_vector = ideal_vector / np.linalg.norm(ideal_vector)
        
        arm_grasps = []
        for i in range(len(gg)):
            grasp = gg[i]
            # 取得夾爪的 Z 軸 (您的代碼中為 [:, 0])
            grasp_z_axis = grasp.rotation_matrix[:, 0] 
            
            alignment_score = np.dot(grasp_z_axis, ideal_vector)
            
            # 💡 淘汰嚴重不順手的姿態 (手腕需要極度扭曲的)
            if alignment_score < -0.2: 
                 continue 
                 
            dist = np.linalg.norm(grasp.translation - arm_base_pos)
            
            # 計算這隻手臂的專屬「順手度分數」
            arm_score = 5.0 + grasp.score + (alignment_score * 3.0) - (dist * 8.0)
            
            arm_grasps.append({
                'orig_idx': i,       # 記錄原本的排序(用來對應顏色)
                'grasp': grasp,      # 原始的 AnyGrasp 物件
                'arm_score': arm_score # 專屬分數
            })

        # 依據這隻手臂的順手程度排序
        arm_grasps.sort(key=lambda x: x['arm_score'], reverse=True)
        return arm_grasps
        # return arm_grasps[:top_k]
    
    # 💡 加入 category 參數，並根據類別動態切換配對策略
    def generate_handover_plan(self, right_grasps, left_grasps, category):
        if not right_grasps or not left_grasps:
            return []
            
        rospy.loginfo(f"🔍 [視覺端大腦] 開始進行左右手專屬姿態配對... 套用策略：【{category}】")
        grouped_pairs = {} 

        # 拿「右手的精選名單」去配「左手的精選名單」
        for r_item in right_grasps:
            for l_item in left_grasps:
                # 確保左右手不會不小心選到同一個點
                if r_item['orig_idx'] == l_item['orig_idx']: 
                    continue 
                
                grasp_R = r_item['grasp']
                grasp_L = l_item['grasp']
                
                pos_R = grasp_R.translation
                pos_L = grasp_L.translation
                
                # 💡 [關鍵修正] 取出夾爪的「前進方向 (X軸)」，計算兩手的相對夾角
                vec_R = grasp_R.rotation_matrix[:, 0] 
                vec_L = grasp_L.rotation_matrix[:, 0]
                dot_approach = np.dot(vec_R, vec_L)
                
                dist = np.linalg.norm(pos_R - pos_L)
                
                score_dist = 0.0
                score_angle = 0.0

                # ==========================================
                # 🤖 依照物件類別進行物理常識過濾與給分
                # ==========================================
                if category == "SYMMETRIC":
                    # 1. 容許 X 軸微幅交叉
                    if pos_R[0] < (pos_L[0] - 0.05): continue 
                    
                    # 2. 防撞底線：確保兩隻手有 6 公分的錯開空間
                    if dist < 0.06: continue 
                    
                    # 3. 🧨 【刪除硬性淘汰】 絕對不要寫 if dot_approach > XXX: continue
                    # 讓所有平行的姿態都能順利活下來！
                    
                    # 4. 💡 軟性加分魔法
                    score_dist = dist * 40.0 
                    
                    # 如果有「正交 (一上一下側)」(dot_approach ≈ 0)：會拿到 (1 - 0) * 10 = 10 分的紅利！
                    # 如果只有「平行 (側面上+側面下)」(dot_approach ≈ 1)：只會拿到 (1 - 1) * 10 = 0 分，但它依然活著！
                    score_angle = (1.0 - dot_approach) * 10.0

                elif category == "LONG":
                    if pos_R[0] < pos_L[0]: continue
                    # 鎚子、香蕉、剪刀：強制握在最遠的兩端！
                    if dist < 0.08: continue 
                    # 工具類通常平行抓，所以不因為 dot_approach > 0.3 而淘汰
                    score_dist = dist * 40.0 # 極度獎勵距離越遠越好
                    score_angle = 0.0 

                elif category == "RIM":
                    # 碗、馬克杯：必須是「面對面」抓取 (向量互相對衝，內積接近 -1)
                    if dist < 0.08: continue 
                    if dot_approach > -0.3: continue # 不是面對面就直接淘汰！
                    score_dist = dist * 10.0 # 距離只要安全就好，權重降低
                    score_angle = (1.0 - dot_approach) * 20.0 # 極度獎勵面對面
                    
                # 保留你原本的高度加分設計
                score_height = pos_R[2] * 10.0 
                
                # 💡 終極總分 = 右手舒服度 + 左手舒服度 + 策略距離分 + 策略角度分 + 高度分
                total_score = r_item['arm_score'] + l_item['arm_score'] + score_dist + score_angle + score_height
                
                pose_R = self.to_ros_pose(grasp_R)
                pose_L = self.to_ros_pose(grasp_L)
                
                r_idx = r_item['orig_idx']
                if r_idx not in grouped_pairs:
                    grouped_pairs[r_idx] = {
                        'orig_idx_R': r_idx,
                        'pose_R': pose_R,
                        'max_score': -float('inf'),
                        'left_candidates': []       
                    }
                
                grouped_pairs[r_idx]['left_candidates'].append({
                    'orig_idx_L': l_item['orig_idx'],
                    'score': total_score,
                    'pose_L': pose_L
                })
                
                if total_score > grouped_pairs[r_idx]['max_score']:
                    grouped_pairs[r_idx]['max_score'] = total_score
                    
        if not grouped_pairs: return []
            
        for i in grouped_pairs:
            grouped_pairs[i]['left_candidates'].sort(key=lambda x: x['score'], reverse=True)
            
        ranked_groups = list(grouped_pairs.values())
        ranked_groups.sort(key=lambda x: x['max_score'], reverse=True)

        json_plan = []
        for group in ranked_groups:
            plan_item = {
                'orig_idx_R': group['orig_idx_R'],
                'max_score': group['max_score'],
                'pose_R_table': self.pose_to_dict(group['pose_R']), 
                'left_candidates': []
            }
            for cand in group['left_candidates']:
                plan_item['left_candidates'].append({
                    'orig_idx_L': cand['orig_idx_L'],
                    'score': cand['score'],
                    'pose_L_table': self.pose_to_dict(cand['pose_L'])
                })
            json_plan.append(plan_item)
            
        return json_plan
            
    def get_arm_base_position(self, arm_name, ref_frame):
        try:
            base_link_name = f"{arm_name}_base_link" 
            trans = self.tf_buffer.lookup_transform(ref_frame, base_link_name, rospy.Time(0), rospy.Duration(1.0))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            rospy.loginfo_once(f"✅ [TF成功] {arm_name} 基座座標: {pos}")
            return pos
        except Exception as e:
            rospy.logwarn_once(f"⚠️ [TF失敗] 使用預設座標。錯誤: {e}")
            if "left" in arm_name: return np.array([-0.5, 0.2, 0.0])
            else: return np.array([0.5, 0.2, 0.0])
            
    # def sort_grasps_by_reachability(self, gg, arm_name, ref_frame):
    #     # 這是您目前提供的版本 (無安全氣囊)
    #     arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        
    #     if "left" in arm_name:
    #         ideal_vector = np.array([1.0, 1.0, 0.0]) 
    #     else:
    #         ideal_vector = np.array([-1.0, 1.0, 0.0])
            
    #     ideal_vector = ideal_vector / np.linalg.norm(ideal_vector)
        
    #     rospy.loginfo(f"篩選中... 手臂: {arm_name} | 理想向量: {ideal_vector}")

    #     reordered_gg = []
        
    #     for i, grasp in enumerate(gg):
    #         # 這裡維持您原本的寫法 grasp_z_axis (實際是指 X 軸)
    #         grasp_z_axis = grasp.rotation_matrix[:, 0] 
            
    #         alignment_score = np.dot(grasp_z_axis, ideal_vector)
    #         dist = np.linalg.norm(grasp.translation - arm_base_pos)
            
    #         original_score = grasp.score
    #         w_align = 3.0   
    #         w_dist = 8.0
    #         base_score = 5.0
            
    #         bonus = 0.0
    #         penalty = 0.0
            
    #         if alignment_score < -0.2: 
    #              grasp.score = -99.0 
    #              final_score = -99.0
    #         else:
    #              bonus = (alignment_score * w_align)
    #              penalty = (dist * w_dist)
    #              final_score = base_score + original_score + bonus - penalty
    #              grasp.score = final_score

    #         grasp.details = {
    #             "base": base_score,
    #             "orig": original_score,
    #             "bonus": bonus,
    #             "penalty": penalty,
    #             "align": alignment_score,
    #             "dist": dist,
    #             "w_dist": w_dist,
    #             "w_align": w_align
    #         }
    #         reordered_gg.append(grasp)

    #     reordered_gg.sort(key=lambda g: g.score, reverse=True)
    #     return reordered_gg
    
    # def callback(self, color_msg, depth_msg):
    #     if not self.need_detection:
    #         return

    #     try:
    #         rospy.loginfo("Processing images...")
            
    #         color_np = imgmsg_to_numpy(color_msg)
    #         if "bgr8" in color_msg.encoding: color_np = color_np[:, :, ::-1]
    #         colors = color_np.astype(np.float32) / 255.0

    #         depths = imgmsg_to_numpy(depth_msg).astype(np.float32)

    #         try: max_depth_val = np.nanmax(depths)
    #         except: max_depth_val = 0.0
    #         scale = 1000.0 if max_depth_val > 100 else 1.0

    #         xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
    #         points_z = depths / scale
    #         points_x = (xmap - self.cx) / self.fx * points_z
    #         points_y = (ymap - self.cy) / self.fy * points_z

    #         mask = (points_z > 0)
    #         points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    #         colors = colors[mask].astype(np.float32)

    #         rospy.loginfo(f"Point Cloud Points: {points.shape[0]}")

    #         # 3. 執行偵測
    #         gg, cloud = self.anygrasp.get_grasp(
    #             points, colors, lims=self.lims, 
    #             apply_object_mask=True, dense_grasp=False, collision_detection=True
    #         )

    #         if len(gg) == 0:
    #             rospy.logwarn("No Grasp detected!")
    #             self.need_detection = False
    #             return

    #         gg = gg.nms().sort_by_score()
            
    #         # 呼叫篩選器
    #         target_arm = self.target_arm_request
    #         camera_frame_id = color_msg.header.frame_id if color_msg.header.frame_id else "camera_color_optical_frame"
    #         gg = self.sort_grasps_by_reachability(gg, target_arm, camera_frame_id)
            
    #         rospy.loginfo(f"Grasps re-sorted for {target_arm} reachability.")

    #         # 【修改】 打包前 5 名 PoseArray
    #         pose_array_msg = PoseArray()
    #         pose_array_msg.header.stamp = rospy.Time.now()
    #         pose_array_msg.header.frame_id = camera_frame_id
            
    #         top_k = min(len(gg), 5)
    #         valid_count = 0
            
    #         print("\n" + "="*40)
    #         print(f"Publishing Top {top_k} Candidates:")
            
    #         for i in range(top_k):
    #             grasp = gg[i]
    #             # 只發布沒被淘汰的 (-99 分)
    #             if grasp.score < 0:
    #                 continue
                
    #             pose = Pose()
    #             pose.position.x = grasp.translation[0]
    #             pose.position.y = grasp.translation[1]
    #             pose.position.z = grasp.translation[2]
                
    #             r = Rotation.from_matrix(grasp.rotation_matrix)
    #             q = r.as_quat()
    #             pose.orientation.x = q[0]
    #             pose.orientation.y = q[1]
    #             pose.orientation.z = q[2]
    #             pose.orientation.w = q[3]
                
    #             pose_array_msg.poses.append(pose)
    #             valid_count += 1
                
    #             print(f"#{i} Score: {grasp.score:.2f} | Pos: {np.round(grasp.translation, 3)}")

    #         print("="*40 + "\n")

    #         if valid_count > 0:
    #             self.grasp_pub.publish(pose_array_msg)
    #             rospy.loginfo(f"PoseArray Published! (Count: {valid_count})")
    #         else:
    #             rospy.logwarn("所有候選姿態都被過濾掉了，未發布任何目標。")

    #         # 5. 視覺化部分 (保持您原本的邏輯，這裡省略以節省版面，請保留原有的視覺化代碼)
    #         if self.cfgs.debug:
    #             print(f"Showing Top {top_k} grasps in 3D...")
    #             print(">>> 顏色圖例: #1=綠色, #2=青色, #3=黃色, #4=紫色, #5=紅色")
    #             print(">>> 請關閉 3D 視窗以繼續程式...")

    #             # 1. 視角轉換
    #             trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                
    #             # 2. 處理點雲
    #             if cloud is None:
    #                 cloud = o3d.geometry.PointCloud()
    #                 cloud.points = o3d.utility.Vector3dVector(points)
    #                 cloud.colors = o3d.utility.Vector3dVector(colors)
    #             cloud.transform(trans_mat)
                
    #             vis_list = [cloud]
                
    #             # 3. 定義顏色表 (RGB)
    #             # 1: Green, 2: Cyan, 3: Yellow, 4: Magenta, 5: Red
    #             rank_colors = [
    #                 [0, 1, 0],   # Rank 1
    #                 [0, 1, 1],   # Rank 2
    #                 [1, 1, 0],   # Rank 3
    #                 [1, 0, 1],   # Rank 4
    #                 [1, 0, 0]    # Rank 5
    #             ]

    #             # 4. 處理前 top_k 個夾爪
    #             for i in range(top_k):
    #                 grasp = gg[i]
    #                 # 如果分數太低(被過濾)，就不顯示或顯示灰色，這裡假設都顯示
                    
    #                 gripper = grasp.to_open3d_geometry()
    #                 gripper.transform(trans_mat)
                    
    #                 # 依照名次上色
    #                 color = rank_colors[i % len(rank_colors)]
    #                 gripper.paint_uniform_color(color)
                    
    #                 vis_list.append(gripper)

    #             # # 5. 加個座標軸方便看方向
    #             # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #             # vis_list.append(axes)

    #             o3d.visualization.draw_geometries(vis_list)

    #         self.need_detection = False

    #     except Exception as e:
    #         rospy.logerr(f"Error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         self.need_detection = False

    def callback(self, color_msg, depth_msg):
        if not self.need_detection: return

        try:
            rospy.loginfo("Processing images...")
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

            gg, cloud = self.anygrasp.get_grasp(
                points, colors, lims=self.lims, 
                apply_object_mask=True, dense_grasp=False, collision_detection=True
            )

            if len(gg) == 0:
                rospy.logwarn("No Grasp detected!")
                self.need_detection = False
                return

            gg = gg.nms().sort_by_score()
            
            # 判斷當前物件類別 (由控制端發送過來的名稱)
            obj_name = self.target_arm_request 
            obj_category = self.categorize_object(obj_name)
            
            top_k = min(len(gg), 10) # 為了能確保左手有足夠的候選點，我們增加到 10
            gg_top = gg[:top_k]

            camera_frame_id = color_msg.header.frame_id if color_msg.header.frame_id else "camera_color_optical_frame"
            
            # 💡 呼叫左右手獨立過濾
            right_grasps = self.get_arm_specific_grasps(gg_top, "rightarm", camera_frame_id)
            left_grasps = self.get_arm_specific_grasps(gg_top, "leftarm", camera_frame_id)
            # right_grasps = self.get_arm_specific_grasps(gg, "rightarm", camera_frame_id, top_k=10)
            # left_grasps = self.get_arm_specific_grasps(gg, "leftarm", camera_frame_id, top_k=10)

            # 💡 進行配對
            json_plan = self.generate_handover_plan(right_grasps, left_grasps, obj_category)
            
            if json_plan:
                plan_msg = String()
                plan_msg.data = json.dumps(json_plan)
                self.plan_pub.publish(plan_msg)
                rospy.loginfo(f"✅ 雙臂交接計畫書 (JSON) 已發布給控制端！(包含 {len(json_plan)} 組右手方案)")
            else:
                rospy.logwarn("視覺端大腦：找不到可行的雙臂交接方案。")

            if self.cfgs.debug:
                # 💡 在這裡專門為畫圖定義要顯示的數量 (例如只畫出 AnyGrasp 原始最高分的前 10 個)
                viz_top_k = min(len(gg), 10)
                
                # print(f"Showing Top {top_k} grasps in 3D...")
                print(f"Showing Top {viz_top_k} grasps in 3D...")
                print(">>> #1=綠色(Green), #2=青色(Cyan),  #3=黃色(Yellow), #4=紫色(Magenta), #5=紅色(Red)")
                print(">>> #6=藍色(Blue),  #7=橘色(Orange),#8=深紫(Purple), #9=深青(Teal),    #10=灰色(Gray)")
                print(">>> 請關閉 3D 視窗以繼續程式...")
                trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                if cloud is None:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points)
                    cloud.colors = o3d.utility.Vector3dVector(colors)
                cloud.transform(trans_mat)
                vis_list = [cloud]
                
                # 因為我們現在取 top_k = 10，我們需要更多顏色來顯示所有點，不然會出錯
                # 這裡擴充了 rank_colors 確保不會 index out of bounds
                rank_colors = [
                    [0, 1, 0],   # Rank 1: Green
                    [0, 1, 1],   # Rank 2: Cyan
                    [1, 1, 0],   # Rank 3: Yellow
                    [1, 0, 1],   # Rank 4: Magenta
                    [1, 0, 0],   # Rank 5: Red
                    [0, 0, 1],   # Rank 6: Blue
                    [1, 0.5, 0], # Rank 7: Orange
                    [0.5, 0, 1], # Rank 8: Purple
                    [0, 0.5, 0.5], # Rank 9: Teal
                    [0.5, 0.5, 0.5] # Rank 10: Gray
                ]
                
                # for i in range(top_k):
                #     grasp = gg_top[i]
                #     gripper = grasp.to_open3d_geometry()
                #     gripper.transform(trans_mat)
                #     color = rank_colors[i % len(rank_colors)]
                #     gripper.paint_uniform_color(color)
                #     vis_list.append(gripper)
                # o3d.visualization.draw_geometries(vis_list)
                
                for i in range(viz_top_k):
                    # 💡 將原本的 gg_top[i] 改成 gg[i]，直接從完整清單拿前幾名來畫
                    grasp = gg[i] 
                    gripper = grasp.to_open3d_geometry()
                    gripper.transform(trans_mat)
                    color = rank_colors[i % len(rank_colors)]
                    gripper.paint_uniform_color(color)
                    vis_list.append(gripper)
                    
                o3d.visualization.draw_geometries(vis_list)

            self.need_detection = False

        except Exception as e:
            rospy.logerr(f"Error: {e}")
            self.need_detection = False

if __name__ == '__main__':
    node = AnyGraspNode()
    # 移除自動觸發，改為純等待
    rospy.spin()