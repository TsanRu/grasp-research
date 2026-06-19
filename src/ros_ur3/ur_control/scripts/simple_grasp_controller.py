#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import os
import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import numpy as np
import json  # 💡 新增 json 模組來解析計畫書
from tf.transformations import quaternion_matrix
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Bool, String
from copy import deepcopy
import threading

import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from tf.transformations import quaternion_from_euler, quaternion_multiply
from tf.transformations import quaternion_matrix, quaternion_from_euler, quaternion_multiply, quaternion_from_matrix
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    
class SimpleGraspController:
    def __init__(self):
        # 1. --- 初始化 MoveIt! & ROS ---
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('p2p_grasp_controller', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # --- 初始化右手 ---
        self.group_name = "rightarm" 
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.planning_frame = self.move_group.get_planning_frame() 
        rospy.loginfo(f"Move Group '{self.group_name}' 初始化完畢。 規劃座標系: {self.planning_frame}")
        
        # --- 初始化左手 ---
        self.left_group_name = "leftarm"
        self.left_move_group = moveit_commander.MoveGroupCommander(self.left_group_name)
        rospy.loginfo(f"Move Group '{self.left_group_name}' 初始化完畢。")
        
        # 2. --- 初始化 TF2 監聽器 (使用你的邏輯) --                                                                                                                                                                                                                                        -
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3. --- 初始化通訊 ---
        self.llm_trigger_pub = rospy.Publisher("/system/trigger_llm", String, queue_size=1)
        self.anygrasp_trigger_pub = rospy.Publisher("/system/trigger_detection", String, queue_size=1)
        # FoundationPose 觸發
        self.fp_trigger_pub = rospy.Publisher(
            "/system/trigger_pose", String, queue_size=1)
        
        #  初始化戰況回報的 Publisher
        self.result_pub = rospy.Publisher("/system/handover_result", String, queue_size=1)
        
        # 4. --- 初始化夾爪 Action Client (右手) ---
        gripper_topic = f'/{self.group_name}/gripper_controller/gripper_cmd'
        self.gripper_client = actionlib.SimpleActionClient(gripper_topic, GripperCommandAction)
        
        rospy.loginfo(f"正在等待右手夾爪伺服器: {gripper_topic} ...")
        if self.gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("右手夾爪伺服器連線成功！")
        else:
            rospy.logwarn("右手夾爪伺服器連線失敗 (Timeout)！")
            self.gripper_client = None

        #  初始化夾爪 Action Client (左手) ---
        left_gripper_topic = f'/{self.left_group_name}/gripper_controller/gripper_cmd'
        self.left_gripper_client = actionlib.SimpleActionClient(left_gripper_topic, GripperCommandAction)
        
        rospy.loginfo(f"正在等待左手夾爪伺服器: {left_gripper_topic} ...")
        if self.left_gripper_client.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("左手夾爪伺服器連線成功！")
        else:
            rospy.logwarn("左手夾爪伺服器連線失敗 (Timeout)！")
            self.left_gripper_client = None
        
        # 5. --- 初始化 Link Attacher 服務 ---
        rospy.loginfo("正在連接 Link Attacher 服務...")
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        try:
            self.attach_srv.wait_for_service(timeout=2.0)
            self.detach_srv.wait_for_service(timeout=2.0)
            rospy.loginfo("Link Attacher 服務連接成功！")
        except rospy.ROSException:
            rospy.logwarn("⚠️ 警告：找不到 /link_attacher_node 服務，物理抓取模擬將無效！")
            
        rospy.sleep(1.0) # 等待連線
        
        # 顏色對照表，方便終端機顯示
        self.COLOR_MAP = {
            0: "綠色 (Rank 1)",
            1: "青色 (Rank 2)",
            2: "黃色 (Rank 3)",
            3: "紫色 (Rank 4)",
            4: "紅色 (Rank 5)",
            5: "藍色 (Rank 6)",
            6: "橘色 (Rank 7)",
            7: "深紫 (Rank 8)",
            8: "深青 (Rank 9)",
            9: "灰色 (Rank 10)"
        }
        
        # 6. --- 設定 MoveIt! 碰撞場景 ---
        self.setup_planning_scene()
        
        self.handover_strategy = "geometric"   # 從 brain 接收
        self.receiver_part = None              # functional_end 時的接取部位名稱
        self.object_points_for_pca = None      # 夾取後暫存物件點雲供 PCA 使用
        self.receiver_centroid_for_pca = None
        self.giver_centroid_for_pca = None
        self.object_centroid_for_pca = None
        self.gc_world_actual = None   # 夾取成功後的實際夾爪位置
        self.object_centroid_offset = None  # 物件中心相對夾爪的偏移
        
        # 待命位置 joint angles
        self.left_standby_joints = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]

        # 直接連到 controller 的 action client（用於同步執行）
        self.right_traj_client = actionlib.SimpleActionClient(
            '/rightarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        self.left_traj_client = actionlib.SimpleActionClient(
            '/leftarm/scaled_pos_joint_traj_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        rospy.loginfo("等待 trajectory action servers...")
        self.right_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        self.left_traj_client.wait_for_server(timeout=rospy.Duration(5.0))
        rospy.loginfo("✅ trajectory action servers 連線成功")
        
        self.latest_points = None
        self.latest_mask = None
        rospy.Subscriber("/anygrasp/object_points", String, self._object_points_cb)

        self.gripper_len = 0.135
        self.right_grasp_center_z = None
        
    # --- 設定規劃場景 (加入桌子) ---
    def setup_planning_scene(self):
        """將 Gazebo 中的環境物件 (如桌子) 加入 MoveIt! 以進行碰撞檢測"""
        rospy.loginfo("正在設定 MoveIt! 規劃場景 (加入障礙物)...")
        
        # 移除舊物件 (避免重複疊加)
        self.scene.remove_world_object("table")
        rospy.sleep(0.5)

        # --- 加入桌子 (參數來自您的第一份程式碼) ---
        table_height = 0.68
        table_size = [2.0, 2.0, table_height]
        
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.planning_frame
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = 1.3 
        table_pose.pose.position.y = 0.0
        # Box 的原點在中心，所以 Z 要設為高度的一半
        table_pose.pose.position.z = table_height / 2.0
        
        self.scene.add_box("table", table_pose, size=table_size)
        
        # 註：不建議在這裡加入 "coke_can" 作為碰撞體，
        # 因為 AnyGrasp 偵測的位置可能跟這裡寫死的不同，會導致 MoveIt 誤判碰撞而無法抓取。
        # 我們只加入靜態環境 (桌子) 即可。

        rospy.loginfo("✅ 桌子已加入場景。")
        rospy.sleep(1.0) # 等待場景更新
        
    # --- 控制夾爪的函式 (提取自 handover.py 並改為 class method) ---
    def control_gripper(self, position, max_effort=1000.0, arm="right"):
        client = self.left_gripper_client if arm == "left" else self.gripper_client
        if client is None:
            rospy.logwarn(f"[{arm}] 夾爪未連線，跳過動作。")
            return

        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        client.send_goal(goal)
        rospy.loginfo(f"[{arm}] 夾爪指令發送: pos={position}")
        
    # --- Attach 物件的函式 ---
    def attach_object(self, object_name, arm="right"):
        """ 呼叫 Gazebo 服務將物件黏在機械手臂上 """
        rospy.loginfo(f"嘗試將物件 '{object_name}' Attach 到 {arm} 手臂...")
        req = AttachRequest()
        req.model_name_1 = object_name      
        req.link_name_1 = "link"            
        req.model_name_2 = "robot"          
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "left" else "rightarm_wrist_3_link"
        
        try:
            self.attach_srv.call(req)
            rospy.loginfo(f"✅ 成功 Attach: {object_name} -> {req.link_name_2}")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Attach 失敗: {e}")
            return False
        
    def detach_object(self, object_name, arm="right"):
        """發生錯誤要放棄抓取時，呼叫此函式解除 Gazebo 裡的物理綁定"""
        rospy.loginfo(f"嘗試解除物件 '{object_name}' 的 Attach 綁定...")
        req = AttachRequest()
        req.model_name_1 = object_name      
        req.link_name_1 = "link"            
        req.model_name_2 = "robot"          
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "left" else "rightarm_wrist_3_link"
        try:
            self.detach_srv.call(req)
            rospy.loginfo("✅ 成功 Detach 釋放物件。")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Detach 失敗: {e}")
            return False
        
    def safe_retreat(self, pose_wrist_grasp, retreat_dist=0.15, arm="right"):
        """
        沿夾取接近方向反向退出，避免撞到物件
        retreat_dist: 退出距離（公尺）
        """
        group = self.left_move_group if arm == "left" else self.move_group
        
        # 取得當前手臂末端姿態
        current_pose = group.get_current_pose().pose
        
        # 從 pose_wrist_grasp 的姿態算出接近方向（tool Z 軸的反方向）
        q = [
            pose_wrist_grasp.orientation.x,
            pose_wrist_grasp.orientation.y,
            pose_wrist_grasp.orientation.z,
            pose_wrist_grasp.orientation.w
        ]
        rot_matrix = quaternion_matrix(q)
        
        # tool Z 軸的反方向（退出方向）
        retreat_vector = -rot_matrix[:3, 2] * retreat_dist
        
        # 從當前位置往退出方向移動
        pose_retreat = deepcopy(current_pose)
        pose_retreat.position.x += retreat_vector[0]
        pose_retreat.position.y += retreat_vector[1]
        pose_retreat.position.z += retreat_vector[2]
        
        rospy.loginfo(f"🔙 [{arm}] 沿接近軸反向退出 {retreat_dist*100:.0f}cm...")
        (plan, fraction) = group.compute_cartesian_path(
            [pose_retreat], 0.01, True
        )
        
        if fraction > 0.8:
            success = group.execute(plan, wait=True)
            if success:
                rospy.loginfo(f"✅ [{arm}] 安全退出成功")
                return True
        
        rospy.logwarn(f"⚠️ [{arm}] 沿軸退出失敗，嘗試垂直向上退出...")
        # 備案：直接往上退
        pose_up = deepcopy(current_pose)
        pose_up.position.z += retreat_dist
        (plan_up, frac_up) = group.compute_cartesian_path(
            [pose_up], 0.01, True
        )
        if frac_up > 0.8:
            group.execute(plan_up, wait=True)
            return True
        
        return False
    
    # --- 安全撤退函式 ---
    def go_home(self, arm="right", pose_near_object=None):
        rospy.loginfo(f"🔄 啟動安全撤退機制：{arm} 退回預設 Home 位置...")
        if pose_near_object is not None:
            self.safe_retreat(pose_near_object, retreat_dist=0.15, arm=arm)
        group = self.left_move_group if arm == "left" else self.move_group
        if arm == "right":
            safe_joint_angles = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
        else:
            safe_joint_angles = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]
        try:
            group.set_joint_value_target(safe_joint_angles)
            success = group.go(wait=True)
            group.stop()
            if success: rospy.loginfo(f"✅ {arm} 已安全撤退至 Home 位置！")
            else: rospy.logwarn(f"⚠️ {arm} 撤退路徑規劃失敗。")
        except Exception as e:
            rospy.logerr(f"❌ 撤退時發生異常: {e}")
            
    # 將 JSON 的 dict 轉回 ROS Pose 結構
    def dict_to_pose(self, d):
        p = Pose()
        p.position.x, p.position.y, p.position.z = d['position']['x'], d['position']['y'], d['position']['z']
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = d['orientation']['x'], d['orientation']['y'], d['orientation']['z'], d['orientation']['w']
        return p
        
    # 改為單點轉換 (因為我們現在是從 JSON 裡一個一個把點拿出來)
    def transform_single_pose(self, pose, source_frame="camera_color_optical_frame"):
        target_frame = self.planning_frame
        p_stamped = PoseStamped()
        p_stamped.header.frame_id = source_frame
        p_stamped.header.stamp = rospy.Time(0)
        p_stamped.pose = pose
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
            p_world = tf2_geometry_msgs.do_transform_pose(p_stamped, transform)
            return p_world.pose
        except Exception as e:
            rospy.logerr(f"TF Transform failed: {e}")
            return None
        
    def _object_points_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.object_points_for_pca = np.array(data["points"])
            rc = data.get("receiver_centroid", None)
            gc = data.get("giver_centroid", None)  
            oc = data.get("object_centroid", None)  
            self.receiver_centroid_for_pca = np.array(rc) if rc is not None else None
            self.giver_centroid_for_pca = np.array(gc) if gc is not None else None
            self.object_centroid_for_pca = np.array(oc) if oc is not None else None
        except Exception:
            pass

    # =========================================================================
    # 在世界座標系下，精準推算左手在空中的交接點
    # =========================================================================
    def calculate_air_handover_pose(self, pose_R_table_world, pose_L_table_world, pose_R_air_world):
            """
            純平移計算：假設右手搬運時不旋轉，直接將 XYZ 移動量套用到左手上。
            """
            # 1. 計算右手在 XYZ 空間中的「移動向量」
            dx = pose_R_air_world.position.x - pose_R_table_world.position.x
            dy = pose_R_air_world.position.y - pose_R_table_world.position.y
            dz = pose_R_air_world.position.z - pose_R_table_world.position.z

            # 2. 複製左手原本在桌上的 Pose (包含原本的 Orientation)
            pose_L_air = deepcopy(pose_L_table_world)
            
            # 3. 將移動向量加到左手的位置上
            pose_L_air.position.x += dx
            pose_L_air.position.y += dy
            pose_L_air.position.z += dz

            # 旋轉 (Orientation) 完全不動，維持 AnyGrasp 找出來的最佳姿勢！
            return pose_L_air
    
    def calculate_wrist_pose(self, grasp_pose, offset_distance):
        """ 
        【移入】 輔助函式：根據指尖位置與夾爪長度，回推手腕位置 
        """
        q = [grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w]
        rot_matrix = quaternion_matrix(q)
        # UR tool0: -Z 是後退方向
        local_offset_vector = np.array([0, 0, -offset_distance, 1])
        global_offset_vector = np.dot(rot_matrix, local_offset_vector)
        
        wrist_pose = deepcopy(grasp_pose)
        wrist_pose.position.x += global_offset_vector[0]
        wrist_pose.position.y += global_offset_vector[1]
        wrist_pose.position.z += global_offset_vector[2]
        return wrist_pose

    def calculate_grasp_targets(self, world_pose):
        """ 
        從 AnyGrasp 的原始 Pose 計算出真正的手腕目標點 
        包含：旋轉修正、Pre-grasp 計算
        """
        # 1. 旋轉修正
        q_orig = [
            world_pose.orientation.x,
            world_pose.orientation.y,
            world_pose.orientation.z,
            world_pose.orientation.w
        ]

        # 依據您之前的代碼進行修正 (抬頭 + 轉手腕)
        q_lift = quaternion_from_euler(0, 1.5708, 0) 
        q_step1 = quaternion_multiply(q_orig, q_lift)
        q_rotate_wrist = quaternion_from_euler(0, 0, -1.5708) 
        q_final = quaternion_multiply(q_step1, q_rotate_wrist)

        # 2. 建立指尖的理想 Pose
        pose_fingertip = deepcopy(world_pose)
        pose_fingertip.orientation.x = q_final[0]
        pose_fingertip.orientation.y = q_final[1]
        pose_fingertip.orientation.z = q_final[2]
        pose_fingertip.orientation.w = q_final[3]
        # pose_fingertip.position.z += 0.01
        
        # 3. 計算手腕位置
        gripper_len = self.gripper_len
        pre_grasp_dist = 0.10   # 預備點距離
        
        # 實際抓取時的手腕位置
        pose_wrist_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len)
        # 預備點的手腕位置
        pose_wrist_pre_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len + pre_grasp_dist)
        
        return pose_fingertip, pose_wrist_pre_grasp, pose_wrist_grasp

    def compute_place_wrist_z(self, arm="left"):
        """根據右手原始夾取高度，動態計算放置時左手的目標 wrist Z。"""
        mg = self.left_move_group if arm == "left" else self.move_group
        current_pose = mg.get_current_pose().pose
        q = [current_pose.orientation.x, current_pose.orientation.y,
             current_pose.orientation.z, current_pose.orientation.w]
        rot = quaternion_matrix(q)
        tool_z_world_z = rot[2, 2]  # 夾爪 Z 軸的世界 Z 分量
        current_grasp_z = current_pose.position.z + self.gripper_len * tool_z_world_z
        delta_z = self.right_grasp_center_z - current_grasp_z
        return current_pose.position.z + delta_z

    def execute_air_handover(self, ranked_groups, object_name,
                         handover_pose, pose_right_table, use_direct_pose=False):
        """
        use_direct_pose=True：receiver_only 模式，姿態已是交接區真實位置，不做平移
        use_direct_pose=False：dual 模式初始結果，需要做平移計算
        """
        self.control_gripper(0.0, arm="left")
        rospy.sleep(0.5)

        total_count = 0
        for i, group in enumerate(ranked_groups):
            candidates = group.get('left_candidates', [])
            if not candidates:
                continue

            for l_count, cand in enumerate(candidates):
                total_count += 1
                rospy.loginfo(f"接收臂嘗試備案 #{total_count}")

                pose_L_camera = self.dict_to_pose(cand['pose_L_table'])
                pose_L_world = self.transform_single_pose(pose_L_camera)
                if pose_L_world is None:
                    continue
                
                rospy.loginfo(
                f"接收臂目標: x={pose_L_world.position.x:.2f}, "
                f"y={pose_L_world.position.y:.2f}, "
                f"z={pose_L_world.position.z:.2f}")

                if use_direct_pose:
                    # receiver_only：直接用偵測到的姿態
                    pose_L_target = pose_L_world
                else:
                    # dual 初始結果：做平移計算
                    pose_L_target = self.calculate_air_handover_pose(
                        pose_right_table, pose_L_world, handover_pose)

                pose_l_fingertip, pose_l_wrist_pre, pose_l_wrist_grasp = \
                    self.calculate_grasp_targets(pose_L_target)

                # Pre-grasp
                self.left_move_group.set_pose_target(pose_l_wrist_pre)
                l_plan_result = self.left_move_group.plan()
                self.left_move_group.clear_pose_targets()

                if not l_plan_result[0]:
                    rospy.logwarn("接收臂 Pre-grasp 規劃失敗，換下一個備案")
                    continue

                if not self.left_move_group.execute(l_plan_result[1], wait=True):
                    self.left_move_group.stop()
                    self.safe_retreat(pose_l_wrist_pre, arm="left")
                    self.go_home("left")
                    continue

                self.left_move_group.stop()

                (plan_l, frac_l) = self.left_move_group.compute_cartesian_path(
                    [pose_l_wrist_grasp], 0.01, True)

                if frac_l < 0.9:
                    rospy.logwarn(f"接收臂 Approach 規劃不完整 ({frac_l:.2f})")
                    self.safe_retreat(pose_l_wrist_pre, arm="left")
                    self.go_home("left")
                    continue

                if not self.left_move_group.execute(plan_l, wait=True):
                    self.left_move_group.stop()
                    self.safe_retreat(pose_l_wrist_grasp, arm="left")
                    self.go_home("left")
                    continue

                self.left_move_group.stop()
                # ── 精度診斷 ──
                rospy.sleep(0.3)
                try:
                    tw = self.tf_buffer.lookup_transform(
                        "world", "leftarm_wrist_3_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    tf = self.tf_buffer.lookup_transform(
                        "world", "leftarm_robotiq_85_left_finger_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    tw_t, tf_t = tw.transform.translation, tf.transform.translation
                    rospy.loginfo(
                        f"📍 [左手] 實際手腕:  x={tw_t.x:.4f}, y={tw_t.y:.4f}, z={tw_t.z:.4f}")
                    rospy.loginfo(
                        f"📍 [左手] 實際指尖:  x={tf_t.x:.4f}, y={tf_t.y:.4f}, z={tf_t.z:.4f}")
                    rospy.loginfo(
                        f"🎯 [左手] 期望指尖:  x={pose_l_fingertip.position.x:.4f}, "
                        f"y={pose_l_fingertip.position.y:.4f}, z={pose_l_fingertip.position.z:.4f}")
                    rospy.loginfo(
                        f"📏 [左手] 指尖誤差:  dx={tf_t.x-pose_l_fingertip.position.x:.4f}, "
                        f"dy={tf_t.y-pose_l_fingertip.position.y:.4f}, "
                        f"dz={tf_t.z-pose_l_fingertip.position.z:.4f}")
                except Exception as e:
                    rospy.logwarn(f"診斷失敗: {e}")
                # ── 診斷結束 ──

                self.control_gripper(0.1, arm="left")
                rospy.sleep(1.0)
                self.attach_object(object_name, arm="left")
                self.detach_object(object_name, arm="right")
                self.control_gripper(0.0, arm="right")

                rospy.loginfo("🎉 交接成功，開始撤退流程...")

                # Step 1（序列）：右手先垂直往上退 15cm，脫離物件空間
                current_r = self.move_group.get_current_pose().pose
                lift_r = deepcopy(current_r)
                lift_r.position.z += 0.15
                (plan_lift, frac_lift) = self.move_group.compute_cartesian_path(
                    [lift_r], 0.01, True)

                if frac_lift >= 0.9:
                    self.move_group.execute(plan_lift, wait=True)
                    self.move_group.stop()
                    rospy.loginfo("✅ 右手垂直上移完成，進入同步撤退")
                else:
                    rospy.logwarn("⚠️ 右手上移規劃失敗，直接序列撤退")
                    self.go_home("right")
                    self.go_home("left")
                    rospy.loginfo("🎉🎉 雙臂空中交接成功！")
                    self.result_pub.publish(json.dumps({
                        "status": "success",
                        "method": "air_handover"
                    }))
                    return True

                # Step 2（同步）：右手回 home + 左手帶物件移到放置區上方
                current_l = self.left_move_group.get_current_pose().pose
                place_above = deepcopy(current_l)
                place_above.position.x = 1.05
                place_above.position.y = -0.3
                place_above.position.z = 0.95

                right_home_joints = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
                self.move_group.set_joint_value_target(right_home_joints)
                plan_right_home = self.move_group.plan()
                self.move_group.clear_pose_targets()

                self.left_move_group.set_pose_target(place_above)
                plan_left_place = self.left_move_group.plan()
                self.left_move_group.clear_pose_targets()

                rospy.loginfo(f"右手 home 規劃: {plan_right_home[0]}")
                rospy.loginfo(f"左手放置區規劃: {plan_left_place[0]}")

                if plan_right_home[0] and plan_left_place[0]:
                    rospy.loginfo("✅ 同步：右手回 home + 左手移到放置區")
                    goal_r = FollowJointTrajectoryGoal()
                    goal_r.trajectory = plan_right_home[1].joint_trajectory
                    goal_l = FollowJointTrajectoryGoal()
                    goal_l.trajectory = plan_left_place[1].joint_trajectory
                    self.right_traj_client.send_goal(goal_r)
                    self.left_traj_client.send_goal(goal_l)
                    self.right_traj_client.wait_for_result()
                    self.left_traj_client.wait_for_result()
                    self.move_group.stop()
                    self.left_move_group.stop()
                    rospy.loginfo("✅ 同步移動完成")

                    # Step 3: 左手動態計算放置高度並放下物件
                    if self.right_grasp_center_z is not None:
                        rospy.loginfo(f"Step 3: 動態放置，基準 grasp_center_z={self.right_grasp_center_z:.4f}")
                        place_down = deepcopy(self.left_move_group.get_current_pose().pose)
                        place_down.position.z = self.compute_place_wrist_z()
                        rospy.loginfo(f"  目標 wrist z={place_down.position.z:.4f}")
                        (plan_down, frac_down) = self.left_move_group.compute_cartesian_path(
                            [place_down], 0.01, True)
                        if frac_down > 0.8:
                            self.left_move_group.execute(plan_down, wait=True)
                            self.left_move_group.stop()
                            self.control_gripper(0.0, arm="left")
                            rospy.sleep(0.5)
                            rospy.loginfo("✅ 物件放置完成")
                        else:
                            rospy.logwarn(f"⚠️ 放置路徑規劃失敗 (frac={frac_down:.2f})，原地釋放")
                            self.control_gripper(0.0, arm="left")
                    else:
                        rospy.logwarn("⚠️ 未記錄右手夾取高度，跳過動態放置，直接釋放")
                        self.control_gripper(0.0, arm="left")

                    self.go_home("left")

                else:
                    rospy.logwarn("⚠️ 同步規劃失敗，序列執行")
                    self.go_home("right")
                    self.go_home("left")

                rospy.loginfo("🎉🎉 雙臂空中交接成功！")
                self.result_pub.publish(json.dumps({
                    "status": "success",
                    "method": "air_handover"
                }))
                return True

        rospy.logwarn("所有接收臂空中備案皆失敗")
        return False


    def execute_left_standalone_grasp(self, left_groups, object_name):
        """
        右手放下物件退開後，左手獨立夾取
        """
        rospy.loginfo("左手開始獨立夾取流程")
        self.control_gripper(0.0, arm="left")
        rospy.sleep(0.5)

        MAX_LEFT_ATTEMPTS = 3

        for i, group in enumerate(left_groups[:MAX_LEFT_ATTEMPTS]):
            rospy.loginfo(f"左手嘗試第 #{i + 1} 組姿態")

            candidates = group.get('left_candidates', [])
            if not candidates:
                continue
            pose_L_camera = self.dict_to_pose(candidates[0]['pose_L_table'])
            pose_L_world = self.transform_single_pose(pose_L_camera)
            if pose_L_world is None:
                continue

            pose_l_fingertip, pose_l_wrist_pre, pose_l_wrist_grasp = \
                self.calculate_grasp_targets(pose_L_world)

            # Pre-grasp
            self.left_move_group.set_pose_target(pose_l_wrist_pre)
            plan_result = self.left_move_group.plan()
            self.left_move_group.clear_pose_targets()

            if not plan_result[0]:
                rospy.logwarn("左手 Pre-grasp 規劃失敗，換下一組")
                continue

            if not self.left_move_group.execute(plan_result[1], wait=True):
                self.left_move_group.stop()
                self.safe_retreat(pose_l_wrist_pre, arm="left")
                self.go_home("left")
                continue

            self.left_move_group.stop()

            # Approach
            (plan_app, frac_app) = self.left_move_group.compute_cartesian_path(
                [pose_l_wrist_grasp], 0.01, True)

            if frac_app < 0.9:
                rospy.logwarn(f"左手 Approach 規劃不完整 ({frac_app:.2f})")
                self.safe_retreat(pose_l_wrist_pre, arm="left")
                self.go_home("left")
                continue

            if not self.left_move_group.execute(plan_app, wait=True):
                self.left_move_group.stop()
                self.safe_retreat(pose_l_wrist_grasp, arm="left")
                self.go_home("left")
                continue

            self.left_move_group.stop()
            self.control_gripper(0.1, arm="left")
            rospy.sleep(1.0)
            self.attach_object(object_name, arm="left")

            rospy.loginfo("🎉 接收臂獨立夾取成功！")
            self.result_pub.publish(json.dumps({
                "status": "success",
                "method": "receiver_standalone"
            }))
            
            return True

        rospy.logerr("左手獨立夾取所有方案皆失敗")
        self.result_pub.publish(json.dumps({"status": "fail"}))
        return False
    
    def request_foundationpose(self, object_name, object_centroid_world=None, timeout=15.0):
        """
        觸發 FoundationPose 估測物件 pose
        
        Returns:
            np.ndarray (4, 4) 物件在相機座標的 pose，或 None
        """
        rospy.loginfo(f"🎯 觸發 FoundationPose 估測 {object_name}...")
        
        result_event = threading.Event()
        result_container = [None]
        
        def fp_cb(msg):
            try:
                result_container[0] = json.loads(msg.data)
                result_event.set()
            except Exception:
                pass
        
        fp_sub = rospy.Subscriber(
            "/pose/foundationpose_result", String, fp_cb)
        rospy.sleep(0.2)
        
        payload = json.dumps({
            "object_name": object_name,
            "object_centroid_world": object_centroid_world.tolist()
                if object_centroid_world is not None else None
        })
        self.fp_trigger_pub.publish(payload)
        
        done = result_event.wait(timeout=timeout)
        fp_sub.unregister()
        
        if not done or result_container[0] is None:
            rospy.logerr("⚠️ FoundationPose 逾時")
            return None
        
        result = result_container[0]
        if result.get("status") != "ok":
            rospy.logerr(f"⚠️ FoundationPose 失敗: {result.get('reason')}")
            return None
        
        pose = np.array(result["pose"]).reshape(4, 4)
        rospy.loginfo(
            f"✓ FoundationPose pose 接收 (translation: {pose[:3,3]})")
        return pose
    
    def trigger_full_detection(self, object_name, mode="dual",
                           rotation_angle=0.0, timeout=240.0):
        
        # # ── receiver_only：跳過 LLM，直接觸發 AnyGrasp ──
        # if mode == "receiver_only":
        #     rospy.loginfo("receiver_only 模式：跳過 LLM，直接觸發 AnyGrasp")
            
        #     plan_event = threading.Event()
        #     plan_container = [None]

        #     def plan_cb(msg):
        #         try:
        #             plan_container[0] = json.loads(msg.data)
        #             plan_event.set()
        #         except Exception:
        #             pass

        #     plan_sub = rospy.Subscriber(
        #         "/anygrasp/handover_plan", String, plan_cb)
        #     rospy.sleep(0.2)

        #     anygrasp_payload = json.dumps({
        #         "object_name": object_name,
        #         "mode": "receiver_only",
        #         "receiver_centroid": self.receiver_centroid_for_pca.tolist()
        #             if self.receiver_centroid_for_pca is not None else None,
        #         "object_centroid": self.object_centroid_for_pca.tolist()
        #             if self.object_centroid_for_pca is not None else None,
        #         "rotation_angle": rotation_angle
        #     })
        #     self.anygrasp_trigger_pub.publish(anygrasp_payload)
            
        #     # FoundationPose 觸發
        #     self.fp_trigger_pub = rospy.Publisher(
        #     "/system/trigger_pose", String, queue_size=1)

        #     plan_done = plan_event.wait(timeout=timeout)
        #     plan_sub.unregister()

        #     if not plan_done or plan_container[0] is None:
        #         rospy.logerr("等待 AnyGrasp receiver_only 回傳逾時")
        #         return None

        #     ranked = plan_container[0]
        #     return ranked if ranked else None
        
        
        # ── receiver_only：先 FP 估 pose → 再觸發 AnyGrasp ──
        if mode == "receiver_only":
            rospy.loginfo("receiver_only 模式：FoundationPose → AnyGrasp")

            # === Step 1: 觸發 FoundationPose ===
            # 推算物件中心在交接區的位置
            oc_handover = None
            if self.object_centroid_offset is not None:
                try:
                    # 使用與 offset 計算時相同的參考點（robotiq_85_base_link）
                    trans_grip_now = self.tf_buffer.lookup_transform(
                        "world", "rightarm_robotiq_85_base_link",
                        rospy.Time(0), rospy.Duration(1.0))
                    t = trans_grip_now.transform.translation
                    gc_now = np.array([t.x, t.y, t.z])
                    oc_handover = gc_now + self.object_centroid_offset
                    rospy.loginfo(
                        f"📍 推算交接區物件中心: {oc_handover.round(3)}")
                except Exception as e:
                    rospy.logwarn(f"⚠️ 無法推算物件中心: {e}")
            fp_pose = self.request_foundationpose(object_name, object_centroid_world=oc_handover)
            fp_pose_list = fp_pose.tolist() if fp_pose is not None else None

            # === Step 2: 觸發 AnyGrasp ===
            plan_event = threading.Event()
            plan_container = [None]

            def plan_cb(msg):
                try:
                    plan_container[0] = json.loads(msg.data)
                    plan_event.set()
                except Exception:
                    pass

            plan_sub = rospy.Subscriber(
                "/anygrasp/handover_plan", String, plan_cb)
            rospy.sleep(0.2)

            anygrasp_payload = json.dumps({
                "object_name": object_name,
                "mode": "receiver_only",
                "receiver_centroid": self.receiver_centroid_for_pca.tolist()
                    if self.receiver_centroid_for_pca is not None else None,
                "object_centroid": self.object_centroid_for_pca.tolist()
                    if self.object_centroid_for_pca is not None else None,
                "rotation_angle": rotation_angle,
                "object_pose_in_cam": fp_pose_list,  # ⭐ 新增
            })
            self.anygrasp_trigger_pub.publish(anygrasp_payload)

            plan_done = plan_event.wait(timeout=timeout)
            plan_sub.unregister()

            if not plan_done or plan_container[0] is None:
                rospy.logerr("等待 AnyGrasp receiver_only 回傳逾時")
                return None

            ranked = plan_container[0]
            return ranked if ranked else None
    
        rospy.loginfo(f"觸發 LLM 前處理 (物件: {object_name}, 模式: {mode})...")
        llm_payload = json.dumps({"object_name": object_name, "mode": mode})

        llm_done_event = threading.Event()
        llm_result_container = [None]

        def llm_done_cb(msg):
            try:
                result = json.loads(msg.data)
                if result.get("status") == "done":
                    llm_result_container[0] = result
                    llm_done_event.set()
            except Exception:
                pass

        llm_sub = rospy.Subscriber("/system/llm_done", String, llm_done_cb)
        rospy.sleep(0.2)
        self.llm_trigger_pub.publish(llm_payload)

        llm_done = llm_done_event.wait(timeout=timeout / 2)
        llm_sub.unregister()

        if not llm_done or llm_result_container[0] is None:
            rospy.logerr("等待 LLM 完成逾時或失敗")
            return None

        rospy.loginfo("LLM 完成，觸發 AnyGrasp...")
        
        llm_data = llm_result_container[0]
        if mode == "dual":  # 只有 dual 才更新策略
            self.handover_strategy = llm_data.get("handover_strategy", "geometric")
            self.receiver_part = llm_data.get("receiver_part", None)
            rospy.loginfo(f"📋 交接策略: {self.handover_strategy}, 接取部位: {self.receiver_part}")

        plan_event = threading.Event()
        plan_container = [None]

        def plan_cb(msg):
            try:
                data = json.loads(msg.data)
                plan_container[0] = data
                plan_event.set()
            except Exception:
                pass

        plan_sub = rospy.Subscriber("/anygrasp/handover_plan", String, plan_cb)
        rospy.sleep(0.2)
        anygrasp_payload = json.dumps({
            "object_name": object_name,
            "mode": mode              # ← 用變數，dual 模式這裡就是 "dual"
        })
        self.anygrasp_trigger_pub.publish(anygrasp_payload)

        plan_done = plan_event.wait(timeout=timeout / 2)
        plan_sub.unregister()

        if not plan_done or plan_container[0] is None:
            rospy.logerr("等待 AnyGrasp 回傳逾時")
            return None

        ranked_groups = plan_container[0]
        rospy.loginfo(f"收到 {len(ranked_groups)} 個候選方案")
        return ranked_groups if len(ranked_groups) > 0 else None
    
    def calculate_rotation_angle_from_pointcloud(self, object_points, receiver_arm_pos, object_pos):
        """
        用 PCA 計算需要旋轉多少度讓物件以正確朝向到達交接區
        object_points: 物件點雲 (Nx3 numpy array，世界座標系)
        receiver_arm_pos: 接收臂基座位置 (世界座標系)
        object_pos: 物件中心位置 (世界座標系)
        """
        if object_points is None or len(object_points) < 10:
            rospy.logwarn("⚠️ 點雲不足，跳過旋轉計算，使用 0 度")
            return 0.0

        centroid = np.mean(object_points, axis=0)
        centered = object_points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 目標方向：物件到接收臂的水平方向
        target_dir = receiver_arm_pos - object_pos
        target_dir[2] = 0
        norm = np.linalg.norm(target_dir)
        if norm < 1e-6:
            return 0.0
        target_dir = target_dir / norm

        if self.handover_strategy == "functional_end":
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

            if self.receiver_centroid_for_pca is not None:
                # 把 receiver_centroid 投影到主軸上
                # 正值表示重心在主軸正方向那端，負值表示在負方向那端
                centroid_to_receiver = self.receiver_centroid_for_pca - centroid
                proj = np.dot(centroid_to_receiver, principal_axis)

                # 讓主軸指向 receiver_centroid 所在的那端（接取端）
                if proj < 0:
                    principal_axis = -principal_axis

                rospy.loginfo(
                    f"✅ 用 receiver_centroid 確認接取端方向，投影值: {proj:.3f}")
            else:
                # fallback：選跟接收臂方向較近的那端
                rospy.logwarn("⚠️ 無 receiver_centroid，改用接收臂方向估算")
                if np.dot(principal_axis[:2], target_dir) < 0:
                    principal_axis = -principal_axis

            current_dir = principal_axis.copy()
            current_dir[2] = 0
            if np.linalg.norm(current_dir) < 1e-6:
                return 0.0
            current_dir = current_dir / np.linalg.norm(current_dir)

        else:  # geometric
            # 找夾爪握得住（寬度 < 8.5cm）且接觸面積最大的面
            gripper_max_width = 0.085
            dims = []
            for i in range(3):
                axis = eigenvectors[:, i]
                proj = centered @ axis
                size = proj.max() - proj.min()
                dims.append((size, eigenvectors[:, i]))

            grippable = [(s, ax) for s, ax in dims if s <= gripper_max_width]
            if grippable:
                grippable.sort(key=lambda x: x[0])
                best_normal = grippable[0][1]
            else:
                dims.sort(key=lambda x: x[0])
                best_normal = dims[0][1]

            current_dir = best_normal.copy()
            current_dir[2] = 0
            if np.linalg.norm(current_dir) < 1e-6:
                return 0.0
            current_dir = current_dir / np.linalg.norm(current_dir)
            if np.dot(current_dir, target_dir) < 0:
                current_dir = -current_dir

        cos_a = np.clip(np.dot(current_dir, target_dir), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_a))
        cross = np.cross(current_dir, target_dir)
        cross_z = cross[2] if hasattr(cross, '__len__') else cross
        if cross_z < 0:
            angle_deg = -angle_deg

        rospy.loginfo(f"PCA 計算旋轉角度: {angle_deg:.1f}°")
        return angle_deg
    
    def calculate_handover_position(self, pose_wrist_grasp_final, current_pose=None):
        grasp_z = pose_wrist_grasp_final.position.z
        handover_z = grasp_z + 0.12

        try:
            trans_r = self.tf_buffer.lookup_transform(
                "world", "rightarm_base_link",
                rospy.Time(0), rospy.Duration(1.0))
            trans_l = self.tf_buffer.lookup_transform(
                "world", "leftarm_base_link",
                rospy.Time(0), rospy.Duration(1.0))
            arm_reach = 0.35
            mid_x = trans_r.transform.translation.x - arm_reach
            mid_y = (trans_r.transform.translation.y
                    + trans_l.transform.translation.y) / 2.0
        except Exception as e:
            rospy.logwarn(f"TF 取得失敗，使用預設值: {e}")
            mid_x, mid_y = 0.425, 0.0

        # X 不動：手臂維持在與夾取相同的 x 距離，僅調整 y 使物件中心對齊兩臂中線
        target_x = pose_wrist_grasp_final.position.x

        if self.object_centroid_offset is not None:
            target_y = mid_y - self.object_centroid_offset[1]
            rospy.loginfo(
                f"📍 物件中心偏移(offset): "
                f"({self.object_centroid_offset[0]:.3f}, {self.object_centroid_offset[1]:.3f})")
        elif self.object_centroid_for_pca is not None:
            delta_y = self.object_centroid_for_pca[1] - pose_wrist_grasp_final.position.y
            target_y = mid_y - delta_y
            rospy.loginfo(
                f"📍 物件中心(點雲fallback): delta_y={delta_y:.3f}")
        elif self.giver_centroid_for_pca is not None and self.receiver_centroid_for_pca is not None:
            obj_center_y = (self.giver_centroid_for_pca[1] + self.receiver_centroid_for_pca[1]) / 2.0
            delta_y = obj_center_y - pose_wrist_grasp_final.position.y
            target_y = mid_y - delta_y
            rospy.loginfo(
                f"📍 物件中心(遮罩fallback): obj_center_y={obj_center_y:.2f}")
        else:
            target_y = pose_wrist_grasp_final.position.y
            rospy.loginfo("📍 物件中心(手腕fallback)")

        rospy.loginfo(f"📍 兩臂中點: ({mid_x:.2f}, {mid_y:.2f})")
        rospy.loginfo(f"📍 右手目標: ({target_x:.2f}, {target_y:.2f}, {handover_z:.2f})")

        handover = Pose()
        handover.position.x = target_x
        handover.position.y = target_y
        handover.position.z = handover_z
        handover.orientation = current_pose.orientation if current_pose is not None \
            else pose_wrist_grasp_final.orientation
        return handover
    
    def rotate_wrist(self, angle_deg, arm="right"):
        group = self.left_move_group if arm == "left" else self.move_group
        
        # 取得當前末端位姿
        current_pose = group.get_current_pose().pose
        
        # 在世界座標系下繞 Z 軸旋轉（不改變 XYZ 位置）
        q_current = [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ]
        # 繞世界 Z 軸的旋轉 quaternion
        q_rot = quaternion_from_euler(0, 0, np.radians(angle_deg))
        # 左乘 = 在世界座標系下旋轉
        q_new = quaternion_multiply(q_rot, q_current)
        
        target_pose = deepcopy(current_pose)
        target_pose.orientation.x = q_new[0]
        target_pose.orientation.y = q_new[1]
        target_pose.orientation.z = q_new[2]
        target_pose.orientation.w = q_new[3]
        
        # 用 Cartesian path 執行（保持位置不變，只改姿態）
        (plan, fraction) = group.compute_cartesian_path(
            [target_pose], 0.01, True)
        
        if fraction > 0.8:
            success = group.execute(plan, wait=True)
            group.stop()
            if success:
                rospy.loginfo(f"✅ 手腕旋轉 {angle_deg:.1f}° 完成（世界 Z 軸）")
                return True
        
        rospy.logwarn(f"⚠️ 手腕旋轉失敗")
        group.stop()
        return False
        
    def execute_mission(self):
        TARGET_OBJECT_NAME = "hammer"
        
        # =========================================================
        # 階段一：視覺偵測，取得右手夾取姿態
        # =========================================================
        ranked_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="dual")
        if ranked_groups is None:
            rospy.logerr("初始偵測失敗，任務中止")
            return

        # =========================================================
        # 階段二：右手夾取（開環，只試一次右手姿態）
        # 改成只試前 N 組，不是無限試錯
        # =========================================================
        grasp_success = False
        pose_wrist_grasp_final = None
        pose_fingertip_final = None
        pose_wrist_pre_final = None
        pose_right_target_final = None

        MAX_RIGHT_ATTEMPTS = 8  # 最多試 3 組右手姿態
        
        for i, group in enumerate(ranked_groups[:MAX_RIGHT_ATTEMPTS]):
            rospy.loginfo(f"右手嘗試第 #{i+1} 組姿態")
            
            pose_R_camera = self.dict_to_pose(group['pose_R_table'])
            pose_right_target = self.transform_single_pose(pose_R_camera)
            if pose_right_target is None:
                continue

            pose_fingertip, pose_wrist_pre, pose_wrist_grasp = \
                self.calculate_grasp_targets(pose_right_target)
            self.right_grasp_center_z = pose_fingertip.position.z

            # Pre-grasp：先規劃，失敗直接換下一組
            self.move_group.set_pose_target(pose_wrist_pre)
            plan_result = self.move_group.plan()
            if not plan_result[0]:
                self.move_group.clear_pose_targets()
                rospy.logwarn("Pre-grasp 規劃失敗，換下一組")
                continue

            # 規劃左手到待命位置，與右手 Pre-grasp 同步執行
            self.left_move_group.set_joint_value_target(self.left_standby_joints)
            plan_standby = self.left_move_group.plan()
            self.left_move_group.clear_pose_targets()

            if plan_standby[0]:
                rospy.loginfo("✅ 同步執行右手 Pre-grasp + 左手待命")
                goal_r = FollowJointTrajectoryGoal()
                goal_r.trajectory = plan_result[1].joint_trajectory
                goal_l = FollowJointTrajectoryGoal()
                goal_l.trajectory = plan_standby[1].joint_trajectory
                self.right_traj_client.send_goal(goal_r)
                self.left_traj_client.send_goal(goal_l)
                self.right_traj_client.wait_for_result()
                self.left_traj_client.wait_for_result()
                self.move_group.stop()
                self.left_move_group.stop()
                rospy.loginfo("✅ 同步移動完成")
            else:
                rospy.logwarn("⚠️ 左手待命規劃失敗，序列執行右手 Pre-grasp")
                if not self.move_group.execute(plan_result[1], wait=True):
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    rospy.logwarn("Pre-grasp 執行失敗，安全退出")
                    self.safe_retreat(pose_wrist_pre, arm="right")
                    self.go_home("right")
                    continue
                self.move_group.stop()
                self.move_group.clear_pose_targets()

            # Approach
            (plan_app, fraction) = self.move_group.compute_cartesian_path(
                [pose_wrist_grasp], 0.01, True)
            if fraction < 0.9 or not self.move_group.execute(plan_app, wait=True):
                self.move_group.stop()
                self.safe_retreat(pose_wrist_grasp, arm="right")
                self.go_home("right")
                continue

            self.move_group.stop()
            
            # ── 精度診斷 ──
            rospy.sleep(0.3)
            try:
                tw = self.tf_buffer.lookup_transform(
                    "world", "rightarm_wrist_3_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tf = self.tf_buffer.lookup_transform(
                    "world", "rightarm_robotiq_85_left_finger_link",
                    rospy.Time(0), rospy.Duration(1.0))
                tw_t, tf_t = tw.transform.translation, tf.transform.translation
                rospy.loginfo(
                    f"📍 [右手] 實際手腕:  x={tw_t.x:.4f}, y={tw_t.y:.4f}, z={tw_t.z:.4f}")
                rospy.loginfo(
                    f"📍 [右手] 實際指尖:  x={tf_t.x:.4f}, y={tf_t.y:.4f}, z={tf_t.z:.4f}")
                rospy.loginfo(
                    f"🎯 [右手] 期望指尖:  x={pose_fingertip.position.x:.4f}, "
                    f"y={pose_fingertip.position.y:.4f}, z={pose_fingertip.position.z:.4f}")
                rospy.loginfo(
                    f"📏 [右手] 指尖誤差:  dx={tf_t.x-pose_fingertip.position.x:.4f}, "
                    f"dy={tf_t.y-pose_fingertip.position.y:.4f}, "
                    f"dz={tf_t.z-pose_fingertip.position.z:.4f}")
            except Exception as e:
                rospy.logwarn(f"診斷失敗: {e}")
            # ── 診斷結束 ──
            
            # 夾取
            self.control_gripper(0.1)
            rospy.sleep(1.0)
            self.attach_object(TARGET_OBJECT_NAME, arm="right")
            
            # 記錄實際夾爪位置，算物件中心相對夾爪的偏移
            try:
                trans_grip = self.tf_buffer.lookup_transform(
                    "world", "rightarm_robotiq_85_base_link",
                    rospy.Time(0), rospy.Duration(1.0))
                t = trans_grip.transform.translation
                self.gc_world_actual = np.array([t.x, t.y, t.z])
                if self.object_centroid_for_pca is not None:
                    self.object_centroid_offset = (
                        self.object_centroid_for_pca - self.gc_world_actual)
                    rospy.loginfo(
                        f"📍 夾爪位置: {self.gc_world_actual.round(3)}")
                    rospy.loginfo(
                        f"📍 物件中心偏移: {self.object_centroid_offset.round(3)}")
            except Exception as e:
                rospy.logwarn(f"⚠️ 無法取得夾爪 TF: {e}")
                self.gc_world_actual = None
                self.object_centroid_offset = None

            # 舉起
            pose_lift = deepcopy(pose_wrist_grasp)
            pose_lift.position.z += 0.12
            (plan_lift, frac_lift) = self.move_group.compute_cartesian_path(
                [pose_lift], 0.01, True)
            if frac_lift < 0.9 or not self.move_group.execute(plan_lift, wait=True):
                self.move_group.stop()
                self.control_gripper(0.0)
                self.detach_object(TARGET_OBJECT_NAME, arm="right")
                self.safe_retreat(pose_wrist_grasp, arm="right")
                self.go_home("right")
                continue

            self.move_group.stop()
            pose_right_target_final = pose_right_target
            grasp_success = True
            pose_wrist_grasp_final = pose_wrist_grasp
            pose_fingertip_final = pose_fingertip
            pose_wrist_pre_final = pose_wrist_pre
            break

        if not grasp_success:
            rospy.logerr("右手夾取失敗，任務中止")
            return

        # =========================================================
        # 階段三：計算旋轉角度並移動到交接區（途中旋轉）
        # =========================================================
        rospy.loginfo("計算物件旋轉角度...")

        # 取得接收臂基座位置
        try:
            trans = self.tf_buffer.lookup_transform(
                "world", "leftarm_base_link", rospy.Time(0), rospy.Duration(1.0))
            receiver_base = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except Exception as e:
            rospy.logwarn(f"無法取得接收臂位置，使用預設值: {e}")
            receiver_base = np.array([0.775, -0.4, 0.69])

        # 物件當前位置（從右手夾取位置估算）
        object_pos = np.array([
            pose_wrist_grasp_final.position.x,
            pose_wrist_grasp_final.position.y,
            pose_wrist_grasp_final.position.z
        ])

        rotation_angle = self.calculate_rotation_angle_from_pointcloud(
            self.object_points_for_pca,
            receiver_base,
            object_pos
        )
        
        # 先旋轉（原地）
        if abs(rotation_angle) > 5.0:
            rospy.loginfo(f"執行手腕旋轉 {rotation_angle:.1f}°...")
            self.rotate_wrist(rotation_angle, arm="right")
            rospy.sleep(0.5)
            # 旋轉後同步更新 offset：物件隨夾爪轉動，世界座標下的偏移向量也跟著旋轉
            if self.object_centroid_offset is not None:
                a = np.radians(rotation_angle)
                Rz = np.array([[np.cos(a), -np.sin(a), 0],
                               [np.sin(a),  np.cos(a), 0],
                               [0,          0,          1]])
                self.object_centroid_offset = Rz @ self.object_centroid_offset
                rospy.loginfo(
                    f"📍 旋轉後更新 offset: {self.object_centroid_offset.round(3)}")

        # 旋轉後再移動到交接區（同時左手移動到待命位置）
        current_pose_after_rotation = self.move_group.get_current_pose().pose
        HANDOVER_POSITION = self.calculate_handover_position(
            pose_wrist_grasp_final, current_pose_after_rotation)

        # 規劃右手軌跡
        (plan_handover, frac_handover) = self.move_group.compute_cartesian_path(
            [HANDOVER_POSITION], 0.01, True)
        if frac_handover < 0.9:
            rospy.logwarn("移動到交接區失敗，放下物件")
            self.control_gripper(0.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="right")
            self.go_home("right")
            return

        # 規劃左手到待命位置的軌跡
        self.left_move_group.set_joint_value_target(self.left_standby_joints)
        plan_standby = self.left_move_group.plan()
        self.left_move_group.clear_pose_targets()

        # 靜態碰撞檢測：確認兩個終點位置不會互相侵入對方工作空間
        right_end_x = HANDOVER_POSITION.position.x
        left_end_joints = self.left_standby_joints
        self.left_move_group.set_joint_value_target(left_end_joints)
        left_end_pose = self.left_move_group.get_current_pose().pose
        self.left_move_group.clear_pose_targets()

        collision_safe = (right_end_x > 0.55) and \
                        plan_standby[0] and \
                        (frac_handover >= 0.9)

        if collision_safe:
            rospy.loginfo("✅ 靜態碰撞檢測通過，同步執行兩臂移動")

            goal_r = FollowJointTrajectoryGoal()
            goal_r.trajectory = plan_handover.joint_trajectory

            goal_l = FollowJointTrajectoryGoal()
            goal_l.trajectory = plan_standby[1].joint_trajectory

            self.right_traj_client.send_goal(goal_r)
            self.left_traj_client.send_goal(goal_l)

            self.right_traj_client.wait_for_result()
            self.left_traj_client.wait_for_result()

            rospy.loginfo(
                f"右手結果: {self.right_traj_client.get_state()}, "
                f"左手結果: {self.left_traj_client.get_state()}")
        else:
            rospy.logwarn("⚠️ 靜態碰撞檢測未通過，改為序列執行")
            if not self.move_group.execute(plan_handover, wait=True):
                self.move_group.stop()
                rospy.logwarn("移動到交接區失敗，放下物件")
                self.control_gripper(0.0)
                self.detach_object(TARGET_OBJECT_NAME, arm="right")
                self.go_home("right")
                return

        self.move_group.stop()
        self.left_move_group.stop()

        # =========================================================
        # 階段四：觸發 receiver_only 重偵測，確認旋轉後的接取姿態
        # =========================================================
        rospy.loginfo("到達交接區，觸發 receiver_only 重偵測...")
        receiver_groups = self.trigger_full_detection(
            TARGET_OBJECT_NAME,
            mode="receiver_only",
            rotation_angle=rotation_angle)

        if receiver_groups is not None and len(receiver_groups) > 0:
            rospy.loginfo("receiver_only 偵測成功，嘗試空中交接")
            success = self.execute_air_handover(
                receiver_groups,
                TARGET_OBJECT_NAME,
                HANDOVER_POSITION,
                pose_right_target_final,
                use_direct_pose=True
            )
            if success:
                return
            rospy.logwarn("空中交接失敗，改走放下重抓流程")
        else:
            rospy.logwarn("receiver_only 偵測失敗，改走放下重抓流程")

        # =========================================================
        # 階段五：空中交接失敗或不適合，放下物件讓左手重新夾取
        # =========================================================
        rospy.loginfo("判定：改走放下重抓流程")
        
        # 取得右手當前末端位置
        current_right_pose = self.move_group.get_current_pose().pose

        # 右手放下物件
        TABLE_HEIGHT = 0.68
        OBJECT_HEIGHT_APPROX = 0.05
        pose_put_down = deepcopy(current_right_pose)
        pose_put_down.position.z = TABLE_HEIGHT + OBJECT_HEIGHT_APPROX

        (plan_down, frac_down) = self.move_group.compute_cartesian_path(
            [pose_put_down], 0.01, True)
        
        if frac_down > 0.5:
            self.move_group.execute(plan_down, wait=True)
            self.move_group.stop()
            self.control_gripper(0.0, arm="right")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="right")
            self.go_home("right", pose_near_object=pose_put_down)
        else:
            # 規劃失敗代表下降路徑有障礙，原地釋放
            rospy.logwarn("無法垂直下降放物件，原地釋放")
            self.control_gripper(0.0, arm="right")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="right")
            self.go_home("right")  # 原地釋放後直接回 Home

        rospy.loginfo("右手已退開，觸發左手重新偵測")
        
        rospy.loginfo("重新觸發 dual 偵測，讓接收臂重新規劃")
        left_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="dual")
        if left_groups is None:
            rospy.logerr("接收臂重新偵測失敗，任務中止")
            return
        self.execute_left_standalone_grasp(left_groups, TARGET_OBJECT_NAME)

if __name__ == '__main__':
    try:
        controller = SimpleGraspController()
        controller.execute_mission()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()