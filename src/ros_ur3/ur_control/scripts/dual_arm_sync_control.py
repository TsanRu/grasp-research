#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.msg import OrientationConstraint
from tf.transformations import quaternion_from_euler
import moveit_msgs
import numpy as np
import pandas as pd
from gazebo_ros_link_attacher.srv import Attach, AttachRequest
import rospy
from gazebo_msgs.srv import SpawnModel
import functools
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetModelState
from moveit_commander import PlanningSceneInterface
import os

class DualArmController:
    def __init__(self):
        rospy.loginfo("--- 🤖 雙臂同步控制器已啟動 ---")

        self.robot = moveit_commander.RobotCommander()
        self.left_arm_group = MoveGroupCommander('leftarm')
        self.right_arm_group = MoveGroupCommander('rightarm')
        self.both_arms_group = MoveGroupCommander('both_arms')
        self.init_attacher()

        self.scene = PlanningSceneInterface()
        rospy.sleep(2)  # 等待 scene 初始化

        # 增加 IK 服務的客戶端，這是更底層、更可靠的 IK 計算方式
        rospy.loginfo("Waiting for IK services...")
        rospy.wait_for_service('/compute_ik')
        self.ik_service = rospy.ServiceProxy('/compute_ik', GetPositionIK)
        rospy.loginfo("IK services are ready.")

        self.both_arms_group.set_planning_time(10)
        rospy.loginfo("已連接到 MoveIt! 規劃組: leftarm, rightarm, both_arms")

    def init_attacher(self):
        rospy.loginfo("等待 Gazebo Attacher 服務...")
        self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
        self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
        self.attach_srv.wait_for_service()
        self.detach_srv.wait_for_service()
        rospy.loginfo("Gazebo Attacher 服務已連線！")

    def move_to_home_joint_space(self):
        rospy.loginfo("--- 準備用 Joint Space 回到 'home' 位置 ---")
        self.both_arms_group.set_named_target("home")
        self.both_arms_group.go(wait=True)

    def move_to_home_cartesian_path(self, arm='both'):
        rospy.loginfo(f"--- 準備用 Cartesian Path 回到 '{arm}' home 位置 ---")
        # 取得目前末端 Pose
        left_current_pose = self.left_arm_group.get_current_pose().pose
        right_current_pose = self.right_arm_group.get_current_pose().pose

        # 設定 home 位置的 Pose（可根據實際 home 姿態調整）
        # 建議先用 set_named_target("home") + go() 讓手臂到 home，然後用 get_current_pose() 取得 home 的 Pose
        left_home_pose = Pose()
        left_home_pose.position.x = 0.9   # 請根據你的 home 位置調整
        left_home_pose.position.y = -0.2
        left_home_pose.position.z = 1.25
        # roll, pitch, yaw = 0, np.pi, 0
        # q = quaternion_from_euler(roll, pitch, yaw)
        left_home_pose.orientation.x = -0.7179520337104552
        left_home_pose.orientation.y = -0.01995893781947982
        left_home_pose.orientation.z = 0.0015368153929035918
        left_home_pose.orientation.w = 0.6958046825730533

        right_home_pose = Pose()
        right_home_pose.position.x = 0.65  # 請根據你的 home 位置調整
        right_home_pose.position.y = 0.2
        right_home_pose.position.z = 1.25
        right_home_pose.orientation.x = -0.01843461122977512
        right_home_pose.orientation.y = -0.7319508088881569
        right_home_pose.orientation.z = 0.6802948793737164
        right_home_pose.orientation.w = 0.03327244467544459

        if arm in ['left', 'both']:
            left_current_pose = self.left_arm_group.get_current_pose().pose
            left_waypoints = [copy.deepcopy(left_current_pose), copy.deepcopy(left_home_pose)]
            (left_plan, left_fraction) = self.left_arm_group.compute_cartesian_path(left_waypoints, 0.01, True)
            # 時間戳補齊
            for idx, point in enumerate(left_plan.joint_trajectory.points):
                point.time_from_start = rospy.Duration(0.1 * idx)
            if left_fraction > 0.95 and left_plan.joint_trajectory.points:
                self.left_arm_group.execute(left_plan, wait=True)
                rospy.loginfo("✅ 左臂已用 Cartesian Path 回到 home。")
            else:
                rospy.logwarn("❌ 左臂 Cartesian Path 規劃失敗，未能回到 home。")
            self.left_arm_group.stop()

        if arm in ['right', 'both']:
            right_current_pose = self.right_arm_group.get_current_pose().pose
            right_waypoints = [copy.deepcopy(right_current_pose), copy.deepcopy(right_home_pose)]
            (right_plan, right_fraction) = self.right_arm_group.compute_cartesian_path(right_waypoints, 0.01, True)
            for idx, point in enumerate(right_plan.joint_trajectory.points):
                point.time_from_start = rospy.Duration(0.1 * idx)
            if right_fraction > 0.95 and right_plan.joint_trajectory.points:
                self.right_arm_group.execute(right_plan, wait=True)
                rospy.loginfo("✅ 右臂已用 Cartesian Path 回到 home。")
            else:
                rospy.logwarn("❌ 右臂 Cartesian Path 規劃失敗，未能回到 home。")
            self.right_arm_group.stop()

    def get_ik_solution(self, group, target_pose):
        """為指定的組和目標姿態計算逆運動學解。"""
        req = GetPositionIKRequest()
        req.ik_request.group_name = group.get_name()
        req.ik_request.pose_stamped.header.frame_id = group.get_planning_frame()
        req.ik_request.pose_stamped.pose = target_pose
        req.ik_request.timeout = rospy.Duration(1.0) # 設置超時

        try:
            resp = self.ik_service.call(req)
            if resp.error_code.val == resp.error_code.SUCCESS:
                return list(resp.solution.joint_state.position[:6]) # 只取前6個關節
            else:
                rospy.logwarn(f"IK for group '{group.get_name()}' failed with error code: {resp.error_code.val}")
                return None
        except rospy.ServiceException as e:
            rospy.logerr(f"IK service call failed: {e}")
            return None

    def move_to_cartesian_targets_sync(self, left_target_pose, right_target_pose):
        rospy.loginfo("--- 準備同步移動雙臂到笛卡爾目標 ---")
        
        # rospy.loginfo("計算左臂的逆運動學解...")
        # left_joint_solution = self.get_ik_solution(self.left_arm_group, left_target_pose)
        
        # rospy.loginfo("計算右臂的逆運動學解...")
        # right_joint_solution = self.get_ik_solution(self.right_arm_group, right_target_pose)

        # # --- 增加除錯日誌 1: 打印 IK 解的結果 ---
        # rospy.loginfo(f"🔍 IK Solution for Left Arm: {left_joint_solution}")
        # rospy.loginfo(f"🔍 IK Solution for Right Arm: {right_joint_solution}")
        # # --- 除錯結束 ---

        left_joint_solution = self.get_joint_solution_from_cartesian_path(self.left_arm_group, left_target_pose)
        right_joint_solution = self.get_joint_solution_from_cartesian_path(self.right_arm_group, right_target_pose)

        if left_joint_solution and right_joint_solution:
            rospy.loginfo("✅ 雙臂的 IK 解都已成功計算。")
            
            combined_joint_target = left_joint_solution + right_joint_solution
            rospy.loginfo(f"設定的12自由度關節目標: {combined_joint_target}")

            # --- 增加除錯日誌 2: 打印手臂當前的關節值 ---
            current_left_joints = self.left_arm_group.get_current_joint_values()
            current_right_joints = self.right_arm_group.get_current_joint_values()
            rospy.loginfo(f"💪 Current Left Joints:  {[f'{j:.3f}' for j in current_left_joints]}")
            rospy.loginfo(f"🎯 Target Left Joints:   {[f'{j:.3f}' for j in left_joint_solution]}")
            rospy.loginfo(f"💪 Current Right Joints: {[f'{j:.3f}' for j in current_right_joints]}")
            rospy.loginfo(f"🎯 Target Right Joints:  {[f'{j:.3f}' for j in right_joint_solution]}")
            # --- 除錯結束 ---
            
            self.both_arms_group.set_joint_value_target(combined_joint_target)
            rospy.loginfo("規劃中...")
            success = self.both_arms_group.go(wait=True)
            
            if success:
                rospy.loginfo("✅ 雙臂已同步移動到目標位置。")
            else:
                rospy.logwarn("❌ 雙臂同步移動規劃或執行失敗。")
                
            self.both_arms_group.stop()
            self.both_arms_group.clear_pose_targets()
        else:
            rospy.logerr("❌ IK 計算失敗，無法移動手臂。")
    
    def get_joint_solution_from_cartesian_path(self, move_group, target_pose):
        """
        【新函數】使用 compute_cartesian_path 來為一個笛卡爾目標尋找一個關節解。
        這比直接調用 IK 服務更穩健，因為它利用了路徑規劃。

        Args:
            move_group (MoveGroupCommander): 要操作的單臂規劃組。
            target_pose (Pose): 目標姿態。

        Returns:
            list or None: 如果規劃成功，返回目標點的關節角度列表；否則返回 None。
        """
        rospy.loginfo(f"--- 為 '{move_group.get_name()}' 使用笛卡爾路徑來計算關節解 ---")
        
        waypoints = [copy.deepcopy(target_pose)]
        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, True)

        if fraction > 0.9 and plan.joint_trajectory.points:
            rospy.loginfo("✅ 笛卡爾路徑計算成功，提取關節解。")
            # 返回規劃路徑的最後一個點的關節角度
            return list(plan.joint_trajectory.points[-1].positions)
        else:
            rospy.logwarn(f"❌ 笛卡爾路徑計算失敗 (覆蓋率: {fraction*100:.1f}%)。無法獲取關節解。")
            return None
        
    def attach_object(self, object_name, object_link, robot_name, robot_link):
        req = AttachRequest()
        req.model_name_1 = object_name
        req.link_name_1 = object_link
        req.model_name_2 = robot_name
        req.link_name_2 = robot_link
        result = self.attach_srv.call(req)
        if result.ok:
            rospy.loginfo(f"已將 {object_name}:{object_link} 附加到 {robot_name}:{robot_link}")
        else:
            rospy.logwarn(f"附加失敗：{object_name} 到 {robot_name}")

    def detach_object(self, object_name, object_link, robot_name, robot_link):
        req = AttachRequest()
        req.model_name_1 = object_name
        req.link_name_1 = object_link
        req.model_name_2 = robot_name
        req.link_name_2 = robot_link
        result = self.detach_srv.call(req)
        if result.ok:
            rospy.loginfo(f"已將 {object_name}:{object_link} 從 {robot_name}:{robot_link} 分離")
        else:
            rospy.logwarn(f"分離失敗：{object_name} 與 {robot_name}")

    def add_ur_base_to_scene(self, name_prefix, base_pose_x, base_pose_y, base_pose_z=0):
        """
        一個輔助函式，用來將一個完整的 ur_base 模型（根據你的 SDF 檔案）加入到 planning scene。
        它會分別加入頂板和四條腿。

        :param name_prefix: 物件的名稱前綴, e.g., "ur_base_left"
        :param base_pose_x: 模型在 world frame 中的 x 座標
        :param base_pose_y: 模型在 world frame 中的 y 座標
        :param base_pose_z: 模型在 world frame 中的 z 座標 (通常為 0)
        """
        rospy.loginfo(f"Adding {name_prefix} to the planning scene...")

        # 1. 頂板 (Top Plate)
        # 根據SDF，頂板相對於模型原點的 pose 是 (0, 0, 0.67501)，尺寸是 (0.5, 0.5, 0.02)
        plate_pose = PoseStamped()
        plate_pose.header.frame_id = "world"
        plate_pose.pose.position.x = base_pose_x + 0.0
        plate_pose.pose.position.y = base_pose_y + 0.0
        plate_pose.pose.position.z = base_pose_z + 0.67501
        self.scene.add_box(f"{name_prefix}_top_plate", plate_pose, size=(0.5, 0.5, 0.02))

        # 定義腿的尺寸和相對位置
        leg_size = (0.04, 0.04, 0.655)
        leg_relative_z = 0.35
        leg_positions = {
            "leg1": (0.22, 0.22),
            "leg2": (-0.22, 0.22),
            "leg3": (0.22, -0.22),
            "leg4": (-0.22, -0.22),
        }

        # 2. 四條腿 (Legs)
        for leg_name, (rel_x, rel_y) in leg_positions.items():
            leg_pose = PoseStamped()
            leg_pose.header.frame_id = "world"
            leg_pose.pose.position.x = base_pose_x + rel_x
            leg_pose.pose.position.y = base_pose_y + rel_y
            leg_pose.pose.position.z = base_pose_z + leg_relative_z
            self.scene.add_box(f"{name_prefix}_{leg_name}", leg_pose, size=leg_size)

    def add_default_scene_objects(self):
        # 桌面 (這個部分保持不變)
        # table_pose = PoseStamped()
        # table_pose.header.frame_id = "world"
        # table_pose.pose.position.x = 0.8
        # table_pose.pose.position.y = 0.0
        # table_pose.pose.position.z = 0.35
        # self.scene.add_box("table", table_pose, size=(1.2, 0.6, 0.7))

        # 地板 (這個部分保持不變)
        # floor_pose = PoseStamped()
        # floor_pose.header.frame_id = "world"
        # floor_pose.pose.position.x = 0.0
        # floor_pose.pose.position.y = 0.0
        # floor_pose.pose.position.z = -0.01
        # self.scene.add_box("floor", floor_pose, size=(5.0, 5.0, 0.02))

        # --- 使用新的輔助函式來加入基座 ---
        # 這些位置是根據你之前 .world 檔案和 Python 程式碼中的設定
        
        # Middle row bases
        self.add_ur_base_to_scene("ur_base_left", 0.775, -0.5)
        self.add_ur_base_to_scene("ur_base_right", 0.775, 0.5)
        self.add_ur_base_to_scene("ur_base_middle", 0.775, 0.0)

        # Top row bases (從你的 .world 檔案)
        self.add_ur_base_to_scene("ur_base_top_left", 0.275, -0.5)
        self.add_ur_base_to_scene("ur_base_top_right", 0.275, 0.5)
        self.add_ur_base_to_scene("ur_base_top_middle", 0.275, 0.0)
        
        # Down row bases (從你的 .world 檔案)
        self.add_ur_base_to_scene("ur_base_down_left", 1.275, -0.5)
        self.add_ur_base_to_scene("ur_base_down_right", 1.275, 0.5)
        self.add_ur_base_to_scene("ur_base_down_middle", 1.275, 0.0)

        rospy.loginfo("已將桌面、地板與所有精確的基座模型加入 Planning Scene")
        rospy.sleep(1)

# test reachable position for dual arm        
def random_reachable_pose_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    reachable_df = df[df['reachable'] == True]
    if reachable_df.empty:
        raise ValueError(f"No reachable points in {csv_path}")
    sample = reachable_df.sample(1).iloc[0]
    pose = Pose()
    pose.position.x = sample['x']
    pose.position.y = sample['y']
    pose.position.z = sample['z']
    # 夾爪朝下
    roll, pitch, yaw = 0, np.pi, 0
    q = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    GRIPPER_DOWN_ORIENTATION = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    rospy.loginfo(f"已定義「夾爪朝下」的姿態: {GRIPPER_DOWN_ORIENTATION}")
    return pose

def move_to_random_reachable_targets_sync(controller, left_csv, right_csv, num_trials=5):
    for i in range(num_trials):
        rospy.loginfo(f"=== 第 {i+1} 次隨機同步移動測試 ===")
        left_pose = random_reachable_pose_from_csv(left_csv)
        right_pose = random_reachable_pose_from_csv(right_csv)
        rospy.loginfo(f"Left target: ({left_pose.position.x:.2f}, {left_pose.position.y:.2f}, {left_pose.position.z:.2f})")
        rospy.loginfo(f"Right target: ({right_pose.position.x:.2f}, {right_pose.position.y:.2f}, {right_pose.position.z:.2f})")
        controller.move_to_cartesian_targets_sync(left_pose, right_pose)
        rospy.sleep(2)

def move_single_arm_cartesian_path(controller, arm, target_xyz, orientation_rpy=(0, np.pi, 0)):
    """
    讓指定手臂以 cartesian path 移動到目標點。
    :param controller: DualArmController 實例
    :param arm: 'left' 或 'right'
    :param target_xyz: 目標座標 (x, y, z) tuple/list
    :param orientation_rpy: 末端姿態 (roll, pitch, yaw)，預設夾爪朝下
    """

    # 選擇對應的 MoveGroup
    if arm == 'left':
        move_group = controller.left_arm_group
    elif arm == 'right':
        move_group = controller.right_arm_group
    else:
        rospy.logwarn("arm 參數請指定 'left' 或 'right'")
        return

    # 取得目前末端 Pose
    current_pose = move_group.get_current_pose().pose

    # 設定目標 Pose
    target_pose = Pose()
    target_pose.position.x = target_xyz[0]
    target_pose.position.y = target_xyz[1]
    target_pose.position.z = target_xyz[2]
    q = quaternion_from_euler(*orientation_rpy)
    target_pose.orientation.x = q[0]
    target_pose.orientation.y = q[1]
    target_pose.orientation.z = q[2]
    target_pose.orientation.w = q[3]

    # Cartesian Path waypoints
    waypoints = [copy.deepcopy(current_pose), copy.deepcopy(target_pose)]

    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, True)
    # 時間戳補齊
    for idx, point in enumerate(plan.joint_trajectory.points):
        point.time_from_start = rospy.Duration(0.1 * idx)

    if fraction > 0.95 and plan.joint_trajectory.points:
        move_group.execute(plan, wait=True)
        rospy.loginfo(f"✅ {arm} 臂已用 Cartesian Path 移動到目標點。")
    else:
        rospy.logwarn(f"❌ {arm} 臂 Cartesian Path 規劃失敗，未能到達目標。")
    move_group.stop()

# assign two specified target position for both arm to move
def move_to_specified_targets_sync(controller, left_xyz, right_xyz, orientation_rpy=(0, np.pi, 0)):
    """
    讓雙手臂同步移動到指定的 XYZ 目標點。
    :param controller: DualArmController 實例
    :param left_xyz: 左手臂目標座標 (x, y, z) tuple/list
    :param right_xyz: 右手臂目標座標 (x, y, z) tuple/list
    :param orientation_rpy: 末端姿態 (roll, pitch, yaw)，預設為夾爪朝下
    """

    # 左手臂目標 Pose
    left_pose = Pose()
    left_pose.position.x = left_xyz[0]
    left_pose.position.y = left_xyz[1]
    left_pose.position.z = left_xyz[2]
    q_left = quaternion_from_euler(*orientation_rpy)
    left_pose.orientation.x = q_left[0]
    left_pose.orientation.y = q_left[1]
    left_pose.orientation.z = q_left[2]
    left_pose.orientation.w = q_left[3]

    # 右手臂目標 Pose
    right_pose = Pose()
    right_pose.position.x = right_xyz[0]
    right_pose.position.y = right_xyz[1]
    right_pose.position.z = right_xyz[2]
    q_right = quaternion_from_euler(*orientation_rpy)
    right_pose.orientation.x = q_right[0]
    right_pose.orientation.y = q_right[1]
    right_pose.orientation.z = q_right[2]
    right_pose.orientation.w = q_right[3]

    # 執行同步移動
    controller.move_to_cartesian_targets_sync(left_pose, right_pose)

    # controller.attach_object("hammer", "link", "robot", "rightarm_wrist_3_link")

def publish_object_pose(model_name, event):
    if rospy.is_shutdown():
        return

    pub = rospy.Publisher("/object_pose", PoseStamped, queue_size=10)

    try:
        # 取得 Gazebo 內物件的位置
        get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        model_state = get_model_state(model_name, "world")

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        pose_msg.pose = model_state.pose

        pub.publish(pose_msg)
        # rospy.loginfo(f"Published {model_name} position: {pose_msg.pose.position}") 

    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to get {model_name} state: {e}")

def spawn_sdf_object(model_name, model_path, x, y, z, roll=0, pitch=0, yaw=0):
    """
    在 Gazebo 中加入 SDF 物件
    :param model_name: 物件名稱
    :param model_path: SDF 檔案路徑
    :param x, y, z: 物件位置
    :param roll, pitch, yaw: 物件旋轉（弧度）
    """
    rospy.wait_for_service("/gazebo/spawn_sdf_model")  # ✅ 確保使用 SDF 服務

    try:
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # 讀取 SDF 檔案
        with open(model_path, "r") as f:
            model_xml = f.read()

        # 設定物件的位置
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        # 設定旋轉
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        # 呼叫 Gazebo 服務
        resp = spawn_model(model_name, model_xml, "/", pose, "world")
        rospy.loginfo(f"Spawned {model_name} at ({x}, {y}, {z})")

        rospy.Timer(
            rospy.Duration(1.0),
            functools.partial(publish_object_pose, model_name)
        )

        rospy.sleep(2)  # 確保 pose 發送啟動

    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to spawn {model_name}: {e}")
    
def main():
    """
    主函數，採用最終的、最穩健的方法，實現真正的雙臂同步移動。
    """
    try:
        # 設定物件資訊
        # object_name = "hammer"  # 你要在 Gazebo 內的名稱
        # object_path = os.path.expanduser("~/ros_ws/src/ros_ur3/ur_gripper_gazebo/models/hammer/model.sdf")  # 替換成你的物件 URDF/SDF 路徑
        # x, y, z = 1.0, 0.13, 0.7  # 物件位置
        # roll, pitch, yaw = 0, 0, 0  # 旋轉角度
        # # add object into gazebo
        # spawn_sdf_object(object_name, object_path, x, y, z, roll, pitch, yaw)

        # dual arm move
        controller = DualArmController()
        controller.add_default_scene_objects()
        
        rospy.loginfo("\n--- 步驟 1: 雙臂同步移動到 Home 位置 ---")
        controller.move_to_home_joint_space()
        rospy.sleep(2)

        left_current_pose = controller.left_arm_group.get_current_pose().pose
        right_current_pose = controller.right_arm_group.get_current_pose().pose

        rospy.loginfo("\n--- 步驟 2: 雙臂同步移動到目標位置，並確保夾爪垂直朝下 ---")
        ## reachable position test
        left_csv = "reachability_map_leftarm.csv"
        right_csv = "reachability_map_rightarm.csv"
        num_trials = 10  # 可自行調整測試次數
        move_to_random_reachable_targets_sync(controller, left_csv, right_csv, num_trials)

        ## assign dual arms target position
        # left_xyz = (0.85, -0.02, 1.02)
        # right_xyz = (1, 0.13, 1.0)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # left_xyz = (0.85, -0.02, 1.02)
        # right_xyz = (1, 0.13, 0.87)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)
        # controller.attach_object("hammer", "link", "robot", "rightarm_wrist_3_link")

        # left_xyz = (0.762, -0.05, 1.00)
        # right_xyz = (1, 0.13, 1.0)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # left_xyz = (0.762, -0.05, 1.1)
        # right_xyz = (0.73, 0.05, 1.0)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # left_xyz = (0.762, -0.05, 1.0)
        # right_xyz = (0.73, 0.05, 1.0)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)
        # controller.attach_object("hammer", "link", "robot", "leftarm_wrist_3_link")
        # controller.detach_object("hammer", "link", "robot", "rightarm_wrist_3_link")

        # left_xyz = (0.762, -0.05, 1.0)
        # right_xyz = (0.73, 0.05, 1.1)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # left_xyz = (0.61, -0.05, 1.0)
        # right_xyz = (0.73, 0.05, 1.1)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # left_xyz = (0.61, -0.05, 0.87)
        # right_xyz = (0.73, 0.05, 1.1)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)
        # controller.detach_object("hammer", "link", "robot", "leftarm_wrist_3_link")

        # left_xyz = (0.61, -0.05, 1.0)
        # right_xyz = (0.73, 0.05, 1.1)
        # move_to_specified_targets_sync(controller, left_xyz, right_xyz)

        # move_single_arm_cartesian_path(controller, 'left', (0.61, -0.05, 1.0))
        # controller.move_to_home_cartesian_path(arm= 'right')

        rospy.loginfo("\n--- 步驟 3: 雙臂同步移動回到 Home 位置 ---")
        controller.move_to_home_joint_space()
        # controller.move_to_home_cartesian_path(arm= 'both')
        
        rospy.loginfo("\n--- 🎉 真正的雙臂同步移動演示完成 ---")

    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("--- 🤖 控制器已關閉 ---")


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('dual_arm_sync_control_node', anonymous=True)
    main()
