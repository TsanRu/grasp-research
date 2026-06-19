#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import moveit_commander
import geometry_msgs.msg
import timeit
import numpy as np

from gazebo_msgs.msg import ModelStates
from ur_control.arm import Arm
from ur_control.constants import GripperType
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

from sensor_msgs.msg import JointState

# Gazebo 與 MoveIt! 的 `base_link` 偏移量與旋轉角度
X_BASE_LINK = 0.11  # Gazebo 內 base_link 的 X 偏移
Y_BASE_LINK = 0.0   # Gazebo 內 base_link 的 Y 偏移
Z_BASE_LINK = 0.69  # Gazebo 內 base_link 的 Z 偏移
YAW = np.pi / 2     # base_link 在 Gazebo 內旋轉 +90° (radians)

def transform_world_to_moveit(x_gaz, y_gaz, z_gaz):
    """
    將 Gazebo 世界座標 (x_gaz, y_gaz, z_gaz) 轉換到
    MoveIt! base_link 座標 (x_moveit, y_moveit, z_moveit)
    """
    # 先扣除 base_link 在 Gazebo 的平移
    x_base = x_gaz - X_BASE_LINK
    y_base = y_gaz - Y_BASE_LINK
    z_base = z_gaz - Z_BASE_LINK

    # 2D 旋轉 +90°（Yaw = +1.57 rad）
    x_moveit = np.cos(YAW) * x_base - np.sin(YAW) * y_base
    y_moveit = np.sin(YAW) * x_base + np.cos(YAW) * y_base
    z_moveit = z_base  # Z 軸不受旋轉影響

    return x_moveit, y_moveit, z_moveit

def get_cube_pose_in_moveit(cube_name):
    """
    從 /gazebo/model_states 取得指定 cube 的座標，
    並轉換到 MoveIt! 的 `base_link` 座標。
    若找不到則回傳 None。
    """
    # 等待 Gazebo 傳送物件座標
    model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)

    # 找到 cube 在 Gazebo 中的索引
    try:
        idx = model_states.name.index(cube_name)
    except ValueError:
        rospy.logerr(f"{cube_name} not found in /gazebo/model_states")
        return None

    # 取得該 cube 在 Gazebo 世界座標的 pose
    cube_pose = model_states.pose[idx]
    x_gaz = cube_pose.position.x
    y_gaz = cube_pose.position.y
    z_gaz = cube_pose.position.z

    # 轉換到 MoveIt! `base_link` 座標
    return transform_world_to_moveit(x_gaz, y_gaz, z_gaz)

def initialize_gripper():

    # 連接到手臂夾爪的 action server
    left_gripper_client = actionlib.SimpleActionClient('/leftarm/gripper_controller/gripper_cmd', GripperCommandAction)
    right_gripper_client = actionlib.SimpleActionClient('/rightarm/gripper_controller/gripper_cmd', GripperCommandAction)

    # 等待 server 啟動
    left_gripper_client.wait_for_server()
    right_gripper_client.wait_for_server()

    return left_gripper_client, right_gripper_client

def pick_and_place_cube(cube_name, target_x, target_y, arm, gripper):
    """
    夾取並搬運指定的方塊 cube_name，並放置在使用者輸入的目標位置。
    arm: MoveGroupCommander("arm")
    gripper: 已經初始化好的夾爪控制物件
    """

    def get_current_pose():
        return arm.get_current_pose().pose

    def move_to_xyz(x, y, z):
        """移動 UR3 TCP (末端) 到指定座標"""
        target_pose = get_current_pose()
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z
        # 設定固定的 orientation，保持 TCP 朝下
        target_pose.orientation.x = -1
        target_pose.orientation.y = 0
        target_pose.orientation.z = 0
        target_pose.orientation.w = 0

        arm.set_pose_target(target_pose)
        arm.go(wait=True)
        arm.stop()
        arm.clear_pose_targets()

    def open_gripper():
        """打開夾爪"""
        gripper.open()

    def close_gripper():
        """關閉夾爪，預設夾取 0.039m 開口距離"""
        gripper.command(0.039)

    # 取得 cube 在 MoveIt! `base_link` 座標系下的位置
    cube_pose_moveit = get_cube_pose_in_moveit(cube_name)
    if cube_pose_moveit is None:
        rospy.logwarn(f"無法找到 {cube_name} 位置，結束此流程。")
        return

    # 展開座標
    cube_x, cube_y, cube_z = cube_pose_moveit
    cube_z += 0.18  # 夾取高度補償
    
    # 設定提升高度（高於物件表面 8cm）
    grasp_height = cube_z + 0.08
    # 設定放置位置高度
    place_height = cube_z + 0.08

    rospy.loginfo(f"[{cube_name}] 位置: x={cube_x:.3f}, y={cube_y:.3f}, z={cube_z:.3f}")
    rospy.loginfo(f"目標位置: x={target_x:.3f}, y={target_y:.3f}")

    # ======= 抓取流程 =======
    # 1. 打開夾爪，移動到方塊正上方
    # open_gripper()
    move_to_xyz(cube_x, cube_y, grasp_height)

    # 2. 下降到抓取位置
    move_to_xyz(cube_x, cube_y, cube_z)

    # 3. 閉合夾爪並在 Gazebo attach 物體
    # close_gripper()
    # gripper.grab(link_name=f"{cube_name}::link")

    # 4. 抬起方塊
    move_to_xyz(cube_x, cube_y, grasp_height)

    # ======= 放置流程 =======
    # 5. 移動到放置目標位置上方
    move_to_xyz(target_x, target_y, place_height)

    # 6. 下降至放置位置
    move_to_xyz(target_x, target_y, cube_z)

    # 7. 放開夾爪並在 Gazebo detach 物體
    # open_gripper()
    # gripper.release(link_name=f"{cube_name}::link")

    # 8. 抬回放置位置上方
    move_to_xyz(target_x, target_y, place_height)

    rospy.loginfo(f"{cube_name} 已放置到目標位置 ({target_x:.3f}, {target_y:.3f})")



def main():
    rospy.init_node('ur3_moveit_control', anonymous=True)
    moveit_commander.roscpp_initialize([])

    # 初始化 MoveIt! Commander
    arm = moveit_commander.MoveGroupCommander("leftarm")
    arm.set_planning_time(5)  # 設定規劃時間
    left_arm_gripper, right_arm_gripper = initialize_gripper()

    # 讓使用者選擇要夾取的方塊
    print("請輸入要夾取的方塊編號 (1, 2, 或 3)：")
    user_input = input("輸入編號：")

    if user_input == "1":
        cube_name = "cube1"
    elif user_input == "2":
        cube_name = "cube2"
    elif user_input == "3":
        cube_name = "cube3"
    else:
        rospy.logwarn("無效的輸入，程式結束。")
        return

    # 讓使用者輸入目標座標 (MoveIt! base_link 下的 X, Y)
    try:
        target_x = float(input("請輸入目標 X 座標 (m): "))
        target_y = float(input("請輸入目標 Y 座標 (m): "))
    except ValueError:
        rospy.logwarn("座標格式錯誤，請輸入數值。程式結束。")
        return

    # 執行夾取並放置
    pick_and_place_cube(cube_name, target_x, target_y, arm, left_arm_gripper)

    # 關閉 MoveIt!
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()