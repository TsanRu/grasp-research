#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np # 確保導入了 numpy

# --- 你提供的 Topics ---
COLOR_TOPIC = "/camera/color/image_raw"
# 注意：根據你的描述，深度 Topic 是 /camera/depth/depth/image_raw
# 一般 Realsense Wrapper 的 topic 可能是 /camera/depth/image_rect_raw
# 請再次用 rostopic list 確認，如果我這裡寫的是對的就不用改
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"

# --- 儲存路徑 ---
# 將圖片儲存到名為 'my_gazebo_data' 的資料夾中
SAVE_DIR = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"

def main():
    """
    主函數，用來訂閱影像話題並儲存圖片。
    """
    # 初始化 ROS 節點
    rospy.init_node('image_saver_for_anygrasp', anonymous=True)
    
    # 建立 CvBridge 實例，用於 ROS Image 和 OpenCV Image 之間的轉換
    bridge = CvBridge()

    # 建立儲存影像的資料夾
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        rospy.loginfo("Created directory: %s", SAVE_DIR)

    # --- 處理彩色影像 ---
    try:
        rospy.loginfo("Waiting for a single color image from topic: %s", COLOR_TOPIC)
        # 等待接收一則訊息，這是一個阻塞操作
        color_msg = rospy.wait_for_message(COLOR_TOPIC, Image, timeout=5.0)
        
        rospy.loginfo("Color image received!")
        
        # 將 ROS Image 訊息轉換為 OpenCV 格式 (BGR8 是 OpenCV 的標準格式)
        cv_color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
        
        # 儲存影像
        color_path = os.path.join(SAVE_DIR, "color.png")
        cv2.imwrite(color_path, cv_color_image)
        
        rospy.loginfo("Color image saved to: %s", color_path)

    except rospy.ROSException as e:
        rospy.logerr("Failed to get color image: %s", e)
        return
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error for color image: %s", e)
        return

    # --- 處理深度影像 ---
    try:
        rospy.loginfo("Waiting for a single depth image from topic: %s", DEPTH_TOPIC)
        depth_msg = rospy.wait_for_message(DEPTH_TOPIC, Image, timeout=5.0)
        rospy.loginfo("Depth image received!")

        # 將 ROS Image 訊息轉換為 OpenCV 格式 (應為 32FC1，代表 32-bit float)
        cv_depth_image_meters = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # === 關鍵修正 ===
        # 1. 將單位從公尺轉換為毫米
        # 2. 將數據類型從 float 轉換為 16-bit unsigned integer (uint16)
        cv_depth_image_mm = (cv_depth_image_meters * 1000).astype(np.uint16)
        
        # 儲存轉換後的 16-bit 毫米單位影像
        depth_path = os.path.join(SAVE_DIR, "depth.png")
        cv2.imwrite(depth_path, cv_depth_image_mm)
        
        rospy.loginfo("Depth image saved to: %s", depth_path)
        rospy.loginfo("Data type of saved depth image: %s", cv_depth_image_mm.dtype)

    except rospy.ROSException as e:
        rospy.logerr("Failed to get depth image: %s", e)
        return
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error for depth image: %s", e)
        return


if __name__ == '__main__':
    main()