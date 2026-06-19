#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np

# 確保能找到 ROS (對齊你原本的設定)
ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

import rospy
from sensor_msgs.msg import Image

def imgmsg_to_numpy(msg):
    """將 ROS Image 轉為 Numpy Array (從你的程式碼移植)"""
    dtype_class = np.uint8
    channels = 1
    if "rgb8" in msg.encoding or "bgr8" in msg.encoding:
        channels = 3
    elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
        dtype_class = np.uint16
    elif "32FC1" in msg.encoding:
        dtype_class = np.float32
    
    img = np.frombuffer(msg.data, dtype=dtype_class)
    if channels > 1:
        img = img.reshape((msg.height, msg.width, channels))
    else:
        img = img.reshape((msg.height, msg.width))
    return img

def save_depth_image():
    # 初始化 ROS 節點
    rospy.init_node('save_depth_node', anonymous=True)
    
    # 根據你的設定，指定 Topic 與存檔資料夾
    topic_name = '/camera/aligned_depth_to_color/image_raw'
    save_dir = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"
    
    # 確保資料夾存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"⏳ 正在等待深度影像 (Topic: {topic_name}) ...")
    
    try:
        # 只抓取一張最新畫面，timeout 設為 10 秒
        msg = rospy.wait_for_message(topic_name, Image, timeout=10.0)
        print("📸 成功擷取到深度畫面！")
        
        # 1. 轉換影像
        depth_np = imgmsg_to_numpy(msg)
        
        # 如果格式是 float32 (公尺)，將其轉換為 uint16 (公釐 mm)，這樣 cv2.imwrite 才能正常存成無損 PNG
        if depth_np.dtype == np.float32:
            depth_np = (depth_np * 1000).astype(np.uint16)
        
        # 2. 儲存原始 16-bit 深度圖 (演算法/AnyGrasp 用)
        raw_path = os.path.join(save_dir, "depth_raw.png")
        cv2.imwrite(raw_path, depth_np)
        print(f"✅ 原始深度圖 (16-bit) 已儲存至: {raw_path}")

        # 3. 儲存人類肉眼可視的 8-bit 熱力圖 (Debug 看爽用)
        vis_path = os.path.join(save_dir, "depth_vis.png")
        # 將深度數值壓縮到 0-255 之間
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 套用 OpenCV 的 JET 偽色彩 (紅暖藍冷)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        cv2.imwrite(vis_path, depth_colormap)
        print(f"✅ 視覺化深度圖 (彩色) 已儲存至: {vis_path}")

    except rospy.ROSException:
        print("❌ 逾時！無法接收到相機畫面，請檢查相機節點或 Gazebo 是否有在跑？")
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")

if __name__ == '__main__':
    save_depth_image()