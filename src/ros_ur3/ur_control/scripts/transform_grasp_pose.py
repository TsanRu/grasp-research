#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs
import tf.transformations as tft  # 新增：用於四元數運算
import numpy as np


def transform_grasp_pose():
    """
    將 AnyGrasp 偵測到的相機座標系下的抓取姿態，轉換為世界座標系。
    """
    rospy.init_node('grasp_pose_transformer')

    # 1. --- 設定 TF2 監聽器 ---
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    # 2. --- 填入你從 AnyGrasp 得到的數據 ---
    source_frame = "camera_depth_optical_frame"
    target_frame = "world"

    grasp_pose_camera = geometry_msgs.msg.PoseStamped()
    grasp_pose_camera.header.frame_id = source_frame
    grasp_pose_camera.header.stamp = rospy.Time(0)

    # 填入位置
    grasp_pose_camera.pose.position.x = 0.2302912
    grasp_pose_camera.pose.position.y = 0.0714927
    grasp_pose_camera.pose.position.z = 0.59952694

    # 填入姿態
    grasp_pose_camera.pose.orientation.x = 0.69438952
    grasp_pose_camera.pose.orientation.y = 0.13350360
    grasp_pose_camera.pose.orientation.z = 0.30725784
    grasp_pose_camera.pose.orientation.w = 0.63686152

    # 3. --- 執行座標轉換 ---
    try:
        rospy.loginfo("Waiting for transform from '%s' to '%s'...", source_frame, target_frame)
        
        # 執行轉換
        grasp_pose_world = tf_buffer.transform(grasp_pose_camera, target_frame, timeout=rospy.Duration(4.0))

        # ============================================
        # 關鍵修正：翻轉 Z 軸方向（方法一）
        # ============================================
        
        # # 提取當前的四元數
        # q_original = [
        #     grasp_pose_world.pose.orientation.x,
        #     grasp_pose_world.pose.orientation.y,
        #     grasp_pose_world.pose.orientation.z,
        #     grasp_pose_world.pose.orientation.w
        # ]
        
        # # 創建一個繞 X 軸旋轉 180 度的四元數
        # # 這會翻轉 Z 軸方向（從向上變向下，或從向下變向上）
        # q_flip = tft.quaternion_from_euler(np.pi, 0, 0)  # (180度, 0, 0)
        
        # # 四元數相乘：先應用原始旋轉，再應用翻轉
        # # 注意順序很重要！
        # q_corrected = tft.quaternion_multiply(q_original, q_flip)
        
        # # 更新姿態
        # grasp_pose_world.pose.orientation.x = q_corrected[0]
        # grasp_pose_world.pose.orientation.y = q_corrected[1]
        # grasp_pose_world.pose.orientation.z = q_corrected[2]
        # grasp_pose_world.pose.orientation.w = q_corrected[3]
        
        # ============================================

        # rospy.loginfo("\n" + "="*50)
        # rospy.loginfo("Successfully Transformed Pose!")
        # rospy.loginfo("="*50)
        # rospy.loginfo("Original Pose (in %s):\n%s", source_frame, grasp_pose_camera.pose)
        # rospy.loginfo("\nTransformed Pose (in %s) - BEFORE correction:\n  Orientation: x=%.4f, y=%.4f, z=%.4f, w=%.4f", 
        #              target_frame, q_original[0], q_original[1], q_original[2], q_original[3])
        # rospy.loginfo("\nCorrected Pose (in %s) - AFTER Z-flip:\n%s", target_frame, grasp_pose_world.pose)
        # rospy.loginfo("="*50)
        
        
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("Successfully Transformed Pose!")
        rospy.loginfo("="*50)
        rospy.loginfo("Original Pose (in %s):\n%s", source_frame, grasp_pose_camera.pose)
        rospy.loginfo("\nTransformed Pose (in %s):\n%s", target_frame, grasp_pose_world.pose)
        rospy.loginfo("="*50)

        return grasp_pose_world

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr("TF Transform failed: %s", e)
        return None


if __name__ == '__main__':
    transform_grasp_pose()
