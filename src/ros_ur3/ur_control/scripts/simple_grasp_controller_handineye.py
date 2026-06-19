#!/usr/bin/env python
# -*- coding: utf-8 -*

import sys
import rospy
import moveit_commander
import tf2_ros
import tf2_geometry_msgs
import geometry_msgs.msg
import numpy as np
import json
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

# ─────────────────────────────────────────────────────────────────────────────
# 術語定義：
#   接收臂 (receiver arm) = 左臂，末端裝有 Hand-in-Eye 相機
#   操作臂 (operator arm) = 右臂，負責從桌面夾取並搬運物件
#
# ROS MoveIt 群組名稱保持原始設定：
#   操作臂 → MoveIt group "rightarm"
#   接收臂 → MoveIt group "leftarm"
# ─────────────────────────────────────────────────────────────────────────────

class SimpleGraspController:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('p2p_grasp_controller', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        # --- 初始化操作臂（rightarm）---
        self.operator_group_name = "rightarm"
        self.operator_group = moveit_commander.MoveGroupCommander(self.operator_group_name)
        self.planning_frame = self.operator_group.get_planning_frame()
        rospy.loginfo(f"操作臂 '{self.operator_group_name}' 初始化完畢。規劃座標系: {self.planning_frame}")

        # --- 初始化接收臂（leftarm）---
        self.receiver_group_name = "leftarm"
        self.receiver_group = moveit_commander.MoveGroupCommander(self.receiver_group_name)
        rospy.loginfo(f"接收臂 '{self.receiver_group_name}' 初始化完畢。")

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.llm_trigger_pub = rospy.Publisher("/system/trigger_llm", String, queue_size=1)
        self.anygrasp_trigger_pub = rospy.Publisher("/system/trigger_detection", String, queue_size=1)
        self.result_pub = rospy.Publisher("/system/handover_result", String, queue_size=1)

        # --- 初始化操作臂夾爪 Action Client ---
        operator_gripper_topic = f'/{self.operator_group_name}/gripper_controller/gripper_cmd'
        self.operator_gripper = actionlib.SimpleActionClient(operator_gripper_topic, GripperCommandAction)

        rospy.loginfo(f"正在等待操作臂夾爪伺服器: {operator_gripper_topic} ...")
        if self.operator_gripper.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("操作臂夾爪伺服器連線成功！")
        else:
            rospy.logwarn("操作臂夾爪伺服器連線失敗 (Timeout)！")
            self.operator_gripper = None

        # --- 初始化接收臂夾爪 Action Client ---
        receiver_gripper_topic = f'/{self.receiver_group_name}/gripper_controller/gripper_cmd'
        self.receiver_gripper = actionlib.SimpleActionClient(receiver_gripper_topic, GripperCommandAction)

        rospy.loginfo(f"正在等待接收臂夾爪伺服器: {receiver_gripper_topic} ...")
        if self.receiver_gripper.wait_for_server(timeout=rospy.Duration(5.0)):
            rospy.loginfo("接收臂夾爪伺服器連線成功！")
        else:
            rospy.logwarn("接收臂夾爪伺服器連線失敗 (Timeout)！")
            self.receiver_gripper = None

        # --- 初始化 Link Attacher 服務 ---
        rospy.loginfo("正在連接 Link Attacher 服務...")
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        try:
            self.attach_srv.wait_for_service(timeout=2.0)
            self.detach_srv.wait_for_service(timeout=2.0)
            rospy.loginfo("Link Attacher 服務連接成功！")
        except rospy.ROSException:
            rospy.logwarn("⚠️ 警告：找不到 /link_attacher_node 服務，物理抓取模擬將無效！")

        rospy.sleep(1.0)

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

        self.setup_planning_scene()

    def setup_planning_scene(self):
        """將 Gazebo 中的環境物件（桌子）加入 MoveIt! 進行碰撞檢測"""
        rospy.loginfo("正在設定 MoveIt! 規劃場景（加入障礙物）...")

        self.scene.remove_world_object("table")
        rospy.sleep(0.5)

        table_height = 0.68
        table_size = [2.0, 2.0, table_height]

        table_pose = PoseStamped()
        table_pose.header.frame_id = self.planning_frame
        table_pose.pose.orientation.w = 1.0
        table_pose.pose.position.x = 1.3
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = table_height / 2.0

        self.scene.add_box("table", table_pose, size=table_size)
        rospy.loginfo("✅ 桌子已加入場景。")
        rospy.sleep(1.0)

    def control_gripper(self, position, max_effort=1000.0, arm="operator"):
        """
        控制夾爪開合
        :param position: 0.0 = 打開, 0.8 = 關閉
        :param arm: "operator"（操作臂）或 "receiver"（接收臂）
        """
        client = self.receiver_gripper if arm == "receiver" else self.operator_gripper
        if client is None:
            rospy.logwarn(f"[{arm}] 夾爪未連線，跳過動作。")
            return

        goal = GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        client.send_goal(goal)
        rospy.loginfo(f"[{arm}] 夾爪指令發送: pos={position}")

    def attach_object(self, object_name, arm="operator"):
        """呼叫 Gazebo 服務將物件黏在手臂上"""
        rospy.loginfo(f"嘗試將物件 '{object_name}' Attach 到 {arm}...")
        req = AttachRequest()
        req.model_name_1 = object_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        # 使用 ROS URDF link 名稱
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "receiver" else "rightarm_wrist_3_link"

        try:
            self.attach_srv.call(req)
            rospy.loginfo(f"✅ 成功 Attach: {object_name} -> {req.link_name_2}")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Attach 失敗: {e}")
            return False

    def detach_object(self, object_name, arm="operator"):
        """解除 Gazebo 物理綁定"""
        rospy.loginfo(f"嘗試解除物件 '{object_name}' 的 Attach 綁定...")
        req = AttachRequest()
        req.model_name_1 = object_name
        req.link_name_1 = "link"
        req.model_name_2 = "robot"
        req.link_name_2 = "leftarm_wrist_3_link" if arm == "receiver" else "rightarm_wrist_3_link"
        try:
            self.detach_srv.call(req)
            rospy.loginfo("✅ 成功 Detach 釋放物件。")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"❌ Detach 失敗: {e}")
            return False

    def safe_retreat(self, pose_wrist_grasp, retreat_dist=0.15, arm="operator"):
        """沿夾取接近方向反向退出，避免撞到物件"""
        group = self.receiver_group if arm == "receiver" else self.operator_group

        current_pose = group.get_current_pose().pose

        q = [
            pose_wrist_grasp.orientation.x,
            pose_wrist_grasp.orientation.y,
            pose_wrist_grasp.orientation.z,
            pose_wrist_grasp.orientation.w
        ]
        rot_matrix = quaternion_matrix(q)

        retreat_vector = -rot_matrix[:3, 2] * retreat_dist

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
        pose_up = deepcopy(current_pose)
        pose_up.position.z += retreat_dist
        (plan_up, frac_up) = group.compute_cartesian_path(
            [pose_up], 0.01, True
        )
        if frac_up > 0.8:
            group.execute(plan_up, wait=True)
            return True

        return False

    def go_home(self, arm="operator", pose_near_object=None):
        rospy.loginfo(f"🔄 啟動安全撤退機制：{arm} 退回預設 Home 位置...")
        if pose_near_object is not None:
            self.safe_retreat(pose_near_object, retreat_dist=0.15, arm=arm)
        group = self.receiver_group if arm == "receiver" else self.operator_group
        if arm == "operator":
            safe_joint_angles = [1.43, -1.211, -2.0, 0.0, 1.6476, -0.0237]
        else:
            safe_joint_angles = [1.5353, -1.211, -1.4186, -0.546, 1.6476, -0.0237]
        try:
            group.set_joint_value_target(safe_joint_angles)
            success = group.go(wait=True)
            group.stop()
            if success:
                rospy.loginfo(f"✅ {arm} 已安全撤退至 Home 位置！")
            else:
                rospy.logwarn(f"⚠️ {arm} 撤退路徑規劃失敗。")
        except Exception as e:
            rospy.logerr(f"❌ 撤退時發生異常: {e}")

    def dict_to_pose(self, d):
        p = Pose()
        p.position.x = d['position']['x']
        p.position.y = d['position']['y']
        p.position.z = d['position']['z']
        p.orientation.x = d['orientation']['x']
        p.orientation.y = d['orientation']['y']
        p.orientation.z = d['orientation']['z']
        p.orientation.w = d['orientation']['w']
        return p

    def transform_single_pose(self, pose, source_frame="camera_color_optical_frame"):
        # Hand-in-Eye 後 AnyGrasp 已轉換到世界座標，直接回傳
        if source_frame == "world":
            return pose
        target_frame = self.planning_frame
        p_stamped = PoseStamped()
        p_stamped.header.frame_id = source_frame
        p_stamped.header.stamp = rospy.Time(0)
        p_stamped.pose = pose
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
            p_world = tf2_geometry_msgs.do_transform_pose(p_stamped, transform)
            return p_world.pose
        except Exception as e:
            rospy.logerr(f"TF Transform failed: {e}")
            return None

    def calculate_air_handover_pose(self, pose_op_table_world, pose_recv_table_world, pose_op_air_world):
        """
        純平移計算：假設操作臂搬運時不旋轉，將 XYZ 移動量套用到接收臂位置上。
        """
        dx = pose_op_air_world.position.x - pose_op_table_world.position.x
        dy = pose_op_air_world.position.y - pose_op_table_world.position.y
        dz = pose_op_air_world.position.z - pose_op_table_world.position.z

        pose_recv_air = deepcopy(pose_recv_table_world)
        pose_recv_air.position.x += dx
        pose_recv_air.position.y += dy
        pose_recv_air.position.z += dz

        return pose_recv_air

    def calculate_wrist_pose(self, grasp_pose, offset_distance):
        """根據指尖位置與夾爪長度，回推手腕位置"""
        q = [grasp_pose.orientation.x, grasp_pose.orientation.y,
             grasp_pose.orientation.z, grasp_pose.orientation.w]
        rot_matrix = quaternion_matrix(q)
        local_offset_vector = np.array([0, 0, -offset_distance, 1])
        global_offset_vector = np.dot(rot_matrix, local_offset_vector)

        wrist_pose = deepcopy(grasp_pose)
        wrist_pose.position.x += global_offset_vector[0]
        wrist_pose.position.y += global_offset_vector[1]
        wrist_pose.position.z += global_offset_vector[2]
        return wrist_pose

    def calculate_grasp_targets(self, world_pose):
        """從 AnyGrasp 的原始 Pose 計算出真正的手腕目標點（含旋轉修正與 Pre-grasp）"""
        q_orig = [
            world_pose.orientation.x,
            world_pose.orientation.y,
            world_pose.orientation.z,
            world_pose.orientation.w
        ]

        q_lift = quaternion_from_euler(0, 1.5708, 0)
        q_step1 = quaternion_multiply(q_orig, q_lift)
        q_rotate_wrist = quaternion_from_euler(0, 0, -1.5708)
        q_final = quaternion_multiply(q_step1, q_rotate_wrist)

        pose_fingertip = deepcopy(world_pose)
        pose_fingertip.orientation.x = q_final[0]
        pose_fingertip.orientation.y = q_final[1]
        pose_fingertip.orientation.z = q_final[2]
        pose_fingertip.orientation.w = q_final[3]

        gripper_len = 0.135
        pre_grasp_dist = 0.10

        pose_wrist_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len)
        pose_wrist_pre_grasp = self.calculate_wrist_pose(pose_fingertip, gripper_len + pre_grasp_dist)

        return pose_fingertip, pose_wrist_pre_grasp, pose_wrist_grasp

    def execute_air_handover(self, ranked_groups, object_name,
                             handover_pose, pose_operator_table):
        """
        Hand-in-Eye 版本：接收臂直接用重新偵測到的姿態執行空中接取
        """
        self.control_gripper(0.0, arm="receiver")
        rospy.sleep(0.5)

        MAX_RECV_ATTEMPTS = 3
        for i, group in enumerate(ranked_groups[:MAX_RECV_ATTEMPTS]):
            rospy.loginfo(f"接收臂嘗試第 #{i+1} 組接取姿態")

            pose_recv_world = self.dict_to_pose(group['pose_R_table'])
            pose_r_fingertip, pose_r_wrist_pre, pose_r_wrist_grasp = \
                self.calculate_grasp_targets(pose_recv_world)

            self.receiver_group.set_pose_target(pose_r_wrist_pre)
            plan_result = self.receiver_group.plan()
            self.receiver_group.clear_pose_targets()

            if not plan_result[0]:
                rospy.logwarn("接收臂 Pre-grasp 規劃失敗，換下一組")
                continue

            if not self.receiver_group.execute(plan_result[1], wait=True):
                self.receiver_group.stop()
                self.safe_retreat(pose_r_wrist_pre, arm="receiver")
                self.go_home("receiver")
                continue

            self.receiver_group.stop()

            (plan_r, frac_r) = self.receiver_group.compute_cartesian_path(
                [pose_r_wrist_grasp], 0.01, True)

            if frac_r < 0.9 or not self.receiver_group.execute(plan_r, wait=True):
                self.receiver_group.stop()
                self.safe_retreat(pose_r_wrist_grasp, arm="receiver")
                self.go_home("receiver")
                continue

            self.receiver_group.stop()
            self.control_gripper(0.1, arm="receiver")
            rospy.sleep(1.0)

            rospy.loginfo("🎉🎉 空中交接成功！")
            self.result_pub.publish(json.dumps({
                "status": "success",
                "method": "air_handover"
            }))
            return True

        rospy.logwarn("所有接收臂接取方案皆失敗")
        return False

    def execute_receiver_standalone_grasp(self, receiver_groups, object_name):
        """
        操作臂放下物件退開後，接收臂獨立夾取
        """
        rospy.loginfo("接收臂開始獨立夾取流程")
        self.control_gripper(0.0, arm="receiver")
        rospy.sleep(0.5)

        MAX_RECV_ATTEMPTS = 3

        for i, group in enumerate(receiver_groups[:MAX_RECV_ATTEMPTS]):
            rospy.loginfo(f"接收臂嘗試第 #{i + 1} 組姿態")

            pose_recv_world = self.dict_to_pose(group['pose_R_table'])
            if pose_recv_world is None:
                continue

            pose_r_fingertip, pose_r_wrist_pre, pose_r_wrist_grasp = \
                self.calculate_grasp_targets(pose_recv_world)

            self.receiver_group.set_pose_target(pose_r_wrist_pre)
            plan_result = self.receiver_group.plan()
            self.receiver_group.clear_pose_targets()

            if not plan_result[0]:
                rospy.logwarn("接收臂 Pre-grasp 規劃失敗，換下一組")
                continue

            if not self.receiver_group.execute(plan_result[1], wait=True):
                self.receiver_group.stop()
                self.safe_retreat(pose_r_wrist_pre, arm="receiver")
                self.go_home("receiver")
                continue

            self.receiver_group.stop()

            (plan_app, frac_app) = self.receiver_group.compute_cartesian_path(
                [pose_r_wrist_grasp], 0.01, True)

            if frac_app < 0.9:
                rospy.logwarn(f"接收臂 Approach 規劃不完整 ({frac_app:.2f})")
                self.safe_retreat(pose_r_wrist_pre, arm="receiver")
                self.go_home("receiver")
                continue

            if not self.receiver_group.execute(plan_app, wait=True):
                self.receiver_group.stop()
                self.safe_retreat(pose_r_wrist_grasp, arm="receiver")
                self.go_home("receiver")
                continue

            self.receiver_group.stop()
            self.control_gripper(0.1, arm="receiver")
            rospy.sleep(1.0)
            self.attach_object(object_name, arm="receiver")

            rospy.loginfo("🎉 接收臂獨立夾取成功！")
            self.result_pub.publish(json.dumps({
                "status": "success",
                "method": "receiver_standalone"
            }))
            return True

        rospy.logerr("接收臂獨立夾取所有方案皆失敗")
        self.result_pub.publish(json.dumps({"status": "fail"}))
        return False

    def trigger_full_detection(self, object_name, mode="dual", timeout=240.0):

        rospy.loginfo(f"觸發 LLM 前處理（物件: {object_name}, 模式: {mode}）...")
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
        self.anygrasp_trigger_pub.publish(object_name)

        plan_done = plan_event.wait(timeout=timeout / 2)
        plan_sub.unregister()

        if not plan_done or plan_container[0] is None:
            rospy.logerr("等待 AnyGrasp 回傳逾時")
            return None

        ranked_groups = plan_container[0]
        rospy.loginfo(f"收到 {len(ranked_groups)} 個候選方案")
        return ranked_groups if len(ranked_groups) > 0 else None

    def go_to_observe_pose(self):
        """
        接收臂移動到初始觀測姿態，使相機俯視桌面工作區。
        ⚠️  這組關節角度需要在實際場景中手動調整：
            - 目標：讓相機能看見桌上物件的擺放區域
            - 同時確保接收臂不遮擋操作臂的工作空間
            建議用 RViz 手動移到理想位置後讀取 /joint_states 填入。
        """
        rospy.loginfo("接收臂移動到觀測位...")
        # 同步 MoveIt 的起始狀態到實際關節位置，避免 START_STATE_IN_COLLISION
        self.receiver_group.set_start_state_to_current_state()
        # 用字典指定名稱，避免與 rqt 字母排序混淆
        # rqt 顯示：elbow / sh_lift / sh_pan / w1 / w2 / w3（字母序）
        # MoveIt 期望：sh_pan / sh_lift / elbow / w1 / w2 / w3（UR 標準序）

        observe_joint_angles = {
            'leftarm_shoulder_pan_joint':   1.6336,
            'leftarm_shoulder_lift_joint': -0.7539,
            'leftarm_elbow_joint':         -1.5707,
            'leftarm_wrist_1_joint':       -1.6336,
            'leftarm_wrist_2_joint':        1.6413,
            'leftarm_wrist_3_joint':       -0.0174,
        }
        try:
            self.receiver_group.set_joint_value_target(observe_joint_angles)
            success = self.receiver_group.go(wait=True)
            self.receiver_group.stop()
            rospy.sleep(0.5)
            return success
        except Exception as e:
            rospy.logerr(f"移動到觀測位失敗: {e}")
            return False

    def execute_mission(self):
        TARGET_OBJECT_NAME = "cracker_box"

        # =========================================================
        # 階段零：接收臂移到觀測位（相機對準桌面工作區）
        # =========================================================
        rospy.loginfo("🔭 接收臂移動到初始觀測位，準備感知場景...")
        if not self.go_to_observe_pose():
            rospy.logerr("接收臂無法到達觀測位，任務中止")
            return
        rospy.loginfo("✅ 接收臂到達觀測位，相機就緒")

        # =========================================================
        # 階段一：視覺偵測，取得操作臂夾取姿態
        # =========================================================
        ranked_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="operator_only")
        if ranked_groups is None:
            rospy.logerr("初始偵測失敗，任務中止")
            return

        # =========================================================
        # 階段二：操作臂夾取（最多試前 N 組姿態）
        # =========================================================
        grasp_success = False
        pose_wrist_grasp_final = None
        pose_fingertip_final = None
        pose_wrist_pre_final = None
        pose_operator_target_final = None

        MAX_OP_ATTEMPTS = 3

        for i, group in enumerate(ranked_groups[:MAX_OP_ATTEMPTS]):
            rospy.loginfo(f"操作臂嘗試第 #{i+1} 組姿態")

            pose_op_camera = self.dict_to_pose(group['pose_R_table'])
            pose_operator_target = self.transform_single_pose(pose_op_camera, source_frame="world")
            if pose_operator_target is None:
                continue

            pose_fingertip, pose_wrist_pre, pose_wrist_grasp = \
                self.calculate_grasp_targets(pose_operator_target)

            # Pre-grasp
            self.operator_group.set_pose_target(pose_wrist_pre)
            plan_result = self.operator_group.plan()
            if not plan_result[0]:
                self.operator_group.clear_pose_targets()
                rospy.logwarn("操作臂 Pre-grasp 規劃失敗，換下一組")
                continue

            if not self.operator_group.execute(plan_result[1], wait=True):
                self.operator_group.stop()
                self.operator_group.clear_pose_targets()
                rospy.logwarn("操作臂 Pre-grasp 執行失敗，安全退出")
                self.safe_retreat(pose_wrist_pre, arm="operator")
                self.go_home("operator")
                continue

            self.operator_group.stop()
            self.operator_group.clear_pose_targets()

            # Approach
            (plan_app, fraction) = self.operator_group.compute_cartesian_path(
                [pose_wrist_grasp], 0.01, True)
            if fraction < 0.9 or not self.operator_group.execute(plan_app, wait=True):
                self.operator_group.stop()
                self.safe_retreat(pose_wrist_grasp, arm="operator")
                self.go_home("operator")
                continue

            self.operator_group.stop()

            # 夾取
            self.control_gripper(0.1, arm="operator")
            rospy.sleep(1.0)
            self.attach_object(TARGET_OBJECT_NAME, arm="operator")

            # 舉起
            pose_lift = deepcopy(pose_wrist_grasp)
            pose_lift.position.z += 0.12
            (plan_lift, frac_lift) = self.operator_group.compute_cartesian_path(
                [pose_lift], 0.01, True)
            if frac_lift < 0.9 or not self.operator_group.execute(plan_lift, wait=True):
                self.operator_group.stop()
                self.control_gripper(0.0, arm="operator")
                self.detach_object(TARGET_OBJECT_NAME, arm="operator")
                self.safe_retreat(pose_wrist_grasp, arm="operator")
                self.go_home("operator")
                continue

            self.operator_group.stop()
            pose_operator_target_final = pose_operator_target
            grasp_success = True
            pose_wrist_grasp_final = pose_wrist_grasp
            pose_fingertip_final = pose_fingertip
            pose_wrist_pre_final = pose_wrist_pre
            break

        if not grasp_success:
            rospy.logerr("操作臂夾取失敗，任務中止")
            return

        # =========================================================
        # 階段三：操作臂移動到固定交接區
        # =========================================================
        HANDOVER_POSITION = Pose()
        HANDOVER_POSITION.position.x = 0.75
        HANDOVER_POSITION.position.y = 0.0
        HANDOVER_POSITION.position.z = 0.95
        HANDOVER_POSITION.orientation = pose_wrist_grasp_final.orientation

        (plan_handover, frac_handover) = self.operator_group.compute_cartesian_path(
            [HANDOVER_POSITION], 0.01, True)
        if frac_handover < 0.9 or not self.operator_group.execute(plan_handover, wait=True):
            self.operator_group.stop()
            rospy.logwarn("操作臂移動到交接區失敗，放下物件")
            self.control_gripper(0.0, arm="operator")
            self.detach_object(TARGET_OBJECT_NAME, arm="operator")
            self.go_home("operator")
            return

        self.operator_group.stop()
        rospy.loginfo("操作臂已到達交接區，準備判斷交接方式")

        # =========================================================
        # 階段四：接收臂移到觀測位，嘗試空中交接
        # =========================================================
        rospy.loginfo("操作臂到位，接收臂移動到觀測位...")
        self.go_to_observe_pose()

        receiver_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="receiver_only")
        if receiver_groups is None:
            rospy.logwarn("接收臂感知失敗，改走放下重抓流程")
        else:
            success = self.execute_air_handover(receiver_groups, TARGET_OBJECT_NAME,
                                                HANDOVER_POSITION, None)
            if success:
                return
            rospy.logwarn("接收臂空中接取失敗，改走放下重抓流程")

        # =========================================================
        # 階段五：空中交接失敗，操作臂放下物件，接收臂重新夾取
        # =========================================================
        rospy.loginfo("判定：改走放下重抓流程")

        current_operator_pose = self.operator_group.get_current_pose().pose

        TABLE_HEIGHT = 0.68
        OBJECT_HEIGHT_APPROX = 0.05
        pose_put_down = deepcopy(current_operator_pose)
        pose_put_down.position.z = TABLE_HEIGHT + OBJECT_HEIGHT_APPROX

        (plan_down, frac_down) = self.operator_group.compute_cartesian_path(
            [pose_put_down], 0.01, True)

        if frac_down > 0.5:
            self.operator_group.execute(plan_down, wait=True)
            self.operator_group.stop()
            self.control_gripper(0.0, arm="operator")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="operator")
            self.go_home("operator", pose_near_object=pose_put_down)
        else:
            rospy.logwarn("操作臂無法垂直下降放物件，原地釋放")
            self.control_gripper(0.0, arm="operator")
            rospy.sleep(1.0)
            self.detach_object(TARGET_OBJECT_NAME, arm="operator")
            self.go_home("operator")

        rospy.loginfo("操作臂已退開，觸發接收臂重新偵測")

        receiver_groups = self.trigger_full_detection(TARGET_OBJECT_NAME, mode="receiver_only")
        if receiver_groups is None:
            rospy.logerr("接收臂重新偵測失敗，任務中止")
            return
        self.execute_receiver_standalone_grasp(receiver_groups, TARGET_OBJECT_NAME)


if __name__ == '__main__':
    try:
        controller = SimpleGraspController()
        controller.execute_mission()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
