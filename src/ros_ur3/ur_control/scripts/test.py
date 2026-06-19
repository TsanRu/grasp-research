#!/usr/bin/env python3
import sys
import numpy as np
import moveit_commander
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import PlanningSceneInterface
from tf.transformations import (quaternion_matrix, quaternion_from_euler,
                                quaternion_multiply, quaternion_from_matrix)
from copy import deepcopy

def calculate_wrist_pose(grasp_pose, offset_distance):
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

def calculate_grasp_targets(world_pose, gripper_len=0.135):
    q_orig = [world_pose.orientation.x, world_pose.orientation.y,
              world_pose.orientation.z, world_pose.orientation.w]
    q_lift = quaternion_from_euler(0, 1.5708, 0)
    q_step1 = quaternion_multiply(q_orig, q_lift)
    q_rotate_wrist = quaternion_from_euler(0, 0, -1.5708)
    q_final = quaternion_multiply(q_step1, q_rotate_wrist)

    pose_fingertip = deepcopy(world_pose)
    pose_fingertip.orientation.x = q_final[0]
    pose_fingertip.orientation.y = q_final[1]
    pose_fingertip.orientation.z = q_final[2]
    pose_fingertip.orientation.w = q_final[3]

    pre_grasp_dist = 0.10
    pose_wrist_grasp = calculate_wrist_pose(pose_fingertip, gripper_len)
    pose_wrist_pre   = calculate_wrist_pose(pose_fingertip, gripper_len + pre_grasp_dist)
    return pose_fingertip, pose_wrist_pre, pose_wrist_grasp

def test_grasp_precision():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('test_grasp_precision', anonymous=True)

    left = moveit_commander.MoveGroupCommander("leftarm")
    scene = PlanningSceneInterface()
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)

    # 加入桌面（與系統設定一致）
    scene.remove_world_object("table")
    rospy.sleep(0.5)
    table_height = 0.68
    table_pose = PoseStamped()
    table_pose.header.frame_id = "world"
    table_pose.pose.orientation.w = 1.0
    table_pose.pose.position.x = 1.3
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = table_height / 2.0
    scene.add_box("table", table_pose, size=[2.0, 2.0, table_height])
    rospy.sleep(1.0)
    rospy.loginfo("✅ 桌面已加入場景")

    # ── Grasp #0 camera frame 資料（從成功 log）──
    # t=[ 0.005 -0.079  0.529]
    # approach(col0)=[ 0.981 -0.193  0.01 ]
    # close(col1)=   [ 0.163  0.851  0.5  ]
    # binormal(col2)=[-0.105 -0.489  0.866]
    rot = np.array([
        [ 0.981,  0.163, -0.105],
        [-0.193,  0.851, -0.489],
        [ 0.01,   0.5,   0.866 ]
    ])
    rot_4x4 = np.eye(4)
    rot_4x4[:3, :3] = rot
    quat = quaternion_from_matrix(rot_4x4)  # [x, y, z, w]

    pose_cam = Pose()
    pose_cam.position.x =  0.005
    pose_cam.position.y = -0.079
    pose_cam.position.z =  0.529
    pose_cam.orientation.x = quat[0]
    pose_cam.orientation.y = quat[1]
    pose_cam.orientation.z = quat[2]
    pose_cam.orientation.w = quat[3]

    rospy.loginfo(f"Camera frame: x={pose_cam.position.x:.4f}, "
                  f"y={pose_cam.position.y:.4f}, z={pose_cam.position.z:.4f}")
    rospy.loginfo(f"Quaternion: x={quat[0]:.4f}, y={quat[1]:.4f}, "
                  f"z={quat[2]:.4f}, w={quat[3]:.4f}")

    # Camera frame → World frame
    p_stamped = PoseStamped()
    p_stamped.header.frame_id = "camera_color_optical_frame"
    p_stamped.header.stamp = rospy.Time(0)
    p_stamped.pose = pose_cam
    try:
        transform = tf_buffer.lookup_transform(
            "world", "camera_color_optical_frame",
            rospy.Time(0), rospy.Duration(4.0))
        pose_world = tf2_geometry_msgs.do_transform_pose(
            p_stamped, transform).pose
    except Exception as e:
        rospy.logerr(f"TF 轉換失敗: {e}")
        return

    rospy.loginfo(f"World frame: x={pose_world.position.x:.4f}, "
                  f"y={pose_world.position.y:.4f}, z={pose_world.position.z:.4f}")

    # 計算夾取目標（gripper_len=0.135，與系統一致）
    pose_fingertip, pose_wrist_pre, pose_wrist_grasp = \
        calculate_grasp_targets(pose_world, gripper_len=0.135)

    rospy.loginfo(f"🎯 期望指尖:   x={pose_fingertip.position.x:.4f}, "
                  f"y={pose_fingertip.position.y:.4f}, "
                  f"z={pose_fingertip.position.z:.4f}")
    rospy.loginfo(f"🎯 手腕 pre:   x={pose_wrist_pre.position.x:.4f}, "
                  f"y={pose_wrist_pre.position.y:.4f}, "
                  f"z={pose_wrist_pre.position.z:.4f}")
    rospy.loginfo(f"🎯 手腕 grasp: x={pose_wrist_grasp.position.x:.4f}, "
                  f"y={pose_wrist_grasp.position.y:.4f}, "
                  f"z={pose_wrist_grasp.position.z:.4f}")

    # Pre-grasp
    left.set_pose_target(pose_wrist_pre)
    plan = left.plan()
    left.clear_pose_targets()
    if not plan[0]:
        rospy.logwarn("❌ Pre-grasp 規劃失敗")
        return
    left.execute(plan[1], wait=True)
    left.stop()
    rospy.loginfo("✅ Pre-grasp 完成")

    # Approach（Cartesian path）
    (plan_app, frac) = left.compute_cartesian_path(
        [pose_wrist_grasp], 0.01, True)
    rospy.loginfo(f"Approach 完整度: {frac:.2f}")
    if frac < 0.9:
        rospy.logwarn("❌ Approach 規劃不完整")
        return
    left.execute(plan_app, wait=True)
    left.stop()
    rospy.loginfo("✅ Approach 完成")

    # 精度診斷
    rospy.sleep(0.3)
    try:
        tw = tf_buffer.lookup_transform(
            "world", "leftarm_wrist_3_link",
            rospy.Time(0), rospy.Duration(1.0))
        tf_ = tf_buffer.lookup_transform(
            "world", "leftarm_robotiq_85_left_finger_link",
            rospy.Time(0), rospy.Duration(1.0))
        tw_t = tw.transform.translation
        tf_t = tf_.transform.translation
        rospy.loginfo(
            f"📍 實際手腕: x={tw_t.x:.4f}, y={tw_t.y:.4f}, z={tw_t.z:.4f}")
        rospy.loginfo(
            f"📍 實際指尖: x={tf_t.x:.4f}, y={tf_t.y:.4f}, z={tf_t.z:.4f}")
        rospy.loginfo(
            f"🎯 期望指尖: x={pose_fingertip.position.x:.4f}, "
            f"y={pose_fingertip.position.y:.4f}, "
            f"z={pose_fingertip.position.z:.4f}")
        rospy.loginfo(
            f"📏 指尖誤差: dx={tf_t.x - pose_fingertip.position.x:.4f}, "
            f"dy={tf_t.y - pose_fingertip.position.y:.4f}, "
            f"dz={tf_t.z - pose_fingertip.position.z:.4f}")
    except Exception as e:
        rospy.logwarn(f"TF 查詢失敗: {e}")

if __name__ == '__main__':
    test_grasp_precision()