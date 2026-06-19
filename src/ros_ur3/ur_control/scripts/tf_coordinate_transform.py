#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs  # 添加這行，註冊PointStamped類型
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
import time

# 初始化 ROS 節點
rospy.init_node('tf_coordinate_transformer', anonymous=True)

# 創建 TF 緩衝區和監聽器
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# 等待 TF 數據可用
rospy.loginfo("等待 TF 數據...")
time.sleep(1.0)  # 給 TF 系統一些時間來接收數據

# 創建一個點（在相機座標系中）
# point_in_camera = PointStamped()
# point_in_camera.header.frame_id = "color"  # 或您的相機座標系名稱
# point_in_camera.header.stamp = rospy.Time.now()
# point_in_camera.point.x = 0.015  # 您系統計算出的 x 座標
# point_in_camera.point.y = 0.095  # 您系統計算出的 y 座標
# point_in_camera.point.z = 0.814  # 您系統計算出的 z 座標


point_in_world = PointStamped()
point_in_world.header.frame_id = "world"  # 或您的相機座標系名稱
point_in_world.header.stamp = rospy.Time.now()
point_in_world.point.x = 1.0  # 您系統計算出的 x 座標
point_in_world.point.y = 0.15  # 您系統計算出的 y 座標
point_in_world.point.z = 0.775  # 您系統計算出的 z 座標

import numpy as np
import tf.transformations

# TF轉換參數 (yaw, pitch, roll)
yaw = -1.5705
pitch = 0
roll = -1.5708

# 計算旋轉矩陣
rotation_matrix = tf.transformations.euler_matrix(roll, pitch, yaw, 'rxyz')[:3, :3]
print("TF轉換計算的旋轉矩陣:")
print(rotation_matrix)

try:
    # 執行轉換
    rospy.loginfo("嘗試轉換座標...")
    # 等待轉換可用
    if tf_buffer.can_transform("world", "color", rospy.Time(0), rospy.Duration(5.0)):
        # point_in_world = tf_buffer.transform(point_in_camera, "world", rospy.Duration(1.0))
        point_in_camera = tf_buffer.transform(point_in_world, "color", rospy.Duration(1.0))
        
        # 現在 point_in_world 包含世界座標系中的座標
        # world_x = point_in_world.point.x
        # world_y = point_in_world.point.y
        # world_z = point_in_world.point.z

        camera_x = point_in_camera.point.x
        camera_y = point_in_camera.point.y
        camera_z = point_in_camera.point.z
        
        # rospy.loginfo(f"物體在世界座標系中的位置: x={world_x}, y={world_y}, z={world_z}")
        rospy.loginfo(f"物體在相機座標系中的位置: x={camera_x}, y={camera_y}, z={camera_z}")
    else:
        rospy.logerr("無法找到從 'rs200_camera' 到 'world' 的轉換")
        
except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    rospy.logerr(f"轉換錯誤: {e}")