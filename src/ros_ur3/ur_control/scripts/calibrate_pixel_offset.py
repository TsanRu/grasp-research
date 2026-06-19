#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import image_geometry
import json
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from ultralytics import YOLO

class PixelOffsetCalibrator:
    def __init__(self, model_path, target_class):
        """
        初始化像素偏差校準器。
        """
        rospy.init_node('pixel_offset_calibrator', anonymous=True)
        
        self.bridge = None
        try:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        except ImportError as e:
            rospy.logerr(f"Failed to import CvBridge: {e}")
            rospy.signal_shutdown("CvBridge not available")
            return

        # ROS 和 TF 相關設置
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 相機模型相關
        self.camera_model = None
        rospy.loginfo("Waiting for camera info...")
        try:
            camera_info_msg = rospy.wait_for_message('//r200/camera/color/camera_info', CameraInfo, timeout=10.0)
            self.camera_info_callback(camera_info_msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Failed to get camera info: {e}")
            rospy.signal_shutdown("Camera info not received")
            return
            
        # YOLO 模型
        self.model = YOLO(model_path)
        self.target_class = target_class
        
        # 數據存儲
        self.latest_rgb_image = None
        self.offset_samples = []
        self.calibration_file = "pixel_offset_calibration.json"

        # 訂閱影像 Topic
        rospy.Subscriber('/r200/camera/color/image_raw', Image, self.image_callback)
        rospy.loginfo("Calibrator initialized. Ready to receive commands.")

    def camera_info_callback(self, msg):
        """處理相機內參訊息，創建相機模型。"""
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)
            rospy.loginfo("Camera model created successfully.")

    def image_callback(self, msg):
        """儲存最新的 RGB 影像。"""
        self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def project_world_to_pixel(self, world_coord):
        """將世界座標投影到 2D 像素座標。"""
        point_world = PointStamped()
        point_world.header.frame_id = "world"
        point_world.header.stamp = rospy.Time(0)
        point_world.point.x, point_world.point.y, point_world.point.z = world_coord

        try:
            # 1. 將世界座標轉換到相機光學座標系
            point_camera = self.tf_buffer.transform(point_world, "color", timeout=rospy.Duration(1.0))
            
            # 2. 將 3D 相機座標投影到 2D 像素平面
            uv_true = self.camera_model.project3dToPixel((point_camera.point.x, point_camera.point.y, point_camera.point.z))
            return uv_true
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF transform failed: {e}")
            return None

    def get_yolo_center(self):
        """在當前影像上運行 YOLO 並返回目標中心點。"""
        if self.latest_rgb_image is None:
            rospy.logwarn("No image received yet.")
            return None

        results = self.model(self.latest_rgb_image, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                if self.model.names[cls_id] == self.target_class:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    u_yolo = (x1 + x2) / 2
                    v_yolo = (y1 + y2) / 2
                    return (u_yolo, v_yolo)
        return None # 如果沒有偵測到目標

    def calibrate(self):
        """主校準循環。"""
        while not rospy.is_shutdown():
            try:
                # 提示用戶輸入
                user_input = input(f"\n👉 請輸入 '{self.target_class}' 的真實世界座標 (x y z)，或按 's' 儲存並退出: ")
                
                if user_input.lower() == 's':
                    if not self.offset_samples:
                        rospy.logwarn("沒有收集到任何樣本，無法儲存。")
                        continue
                    # 計算並儲存
                    avg_offset = np.mean(self.offset_samples, axis=0)
                    rospy.loginfo(f"📊 計算出的平均像素偏差為: (dx: {avg_offset[0]:.2f}, dy: {avg_offset[1]:.2f})")
                    self.save_calibration(avg_offset)
                    break

                # 解析用戶輸入
                parts = user_input.split()
                if len(parts) != 3:
                    rospy.logwarn("輸入格式錯誤，請輸入三個由空格分隔的數字。")
                    continue
                
                world_coord = tuple(map(float, parts))
                rospy.loginfo(f"📍 設定的地面實況座標: {world_coord}")

                # 獲取 YOLO 中心點
                rospy.loginfo("🔍 正在偵測物件...")
                uv_yolo = self.get_yolo_center()
                if uv_yolo is None:
                    rospy.logerr(f"無法在當前畫面中偵測到 '{self.target_class}'，請確保物件可見。")
                    continue
                rospy.loginfo(f"👁️ YOLO BBox 中心點: (u: {uv_yolo[0]:.2f}, v: {uv_yolo[1]:.2f})")

                # 投影真實座標
                uv_true = self.project_world_to_pixel(world_coord)
                if uv_true is None:
                    rospy.logerr("無法完成世界座標到像素座標的投影。")
                    continue
                rospy.loginfo(f"🎯 真實質心投影點: (u: {uv_true[0]:.2f}, v: {uv_true[1]:.2f})")

                # 計算並記錄偏差
                offset = (uv_yolo[0] - uv_true[0], uv_yolo[1] - uv_true[1])
                self.offset_samples.append(offset)
                rospy.loginfo(f"✅ 單次像素偏差: (dx: {offset[0]:.2f}, dy: {offset[1]:.2f})")
                rospy.loginfo(f"已收集 {len(self.offset_samples)} 個樣本。")

            except Exception as e:
                rospy.logerr(f"發生錯誤: {e}")

    def save_calibration(self, avg_offset):
        """將校準結果儲存到 JSON 文件。"""
        data = {
            'target_class': self.target_class,
            'pixel_offset': list(avg_offset)
        }
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f"校準數據已成功儲存到 {self.calibration_file}")

if __name__ == '__main__':
    # --- 請在這裡修改您的模型路徑和目標類別名稱 ---
    YOLO_MODEL_PATH = 'object_dataset/v5/runs/detect/yolo11_custom_training/weights/best.pt'
    TARGET_CLASS_NAME = '-hammer-' # 與您模型中的類別名稱完全一致
    # ---------------------------------------------
    
    try:
        calibrator = PixelOffsetCalibrator(YOLO_MODEL_PATH, TARGET_CLASS_NAME)
        calibrator.calibrate()
    except rospy.ROSInterruptException:
        pass
