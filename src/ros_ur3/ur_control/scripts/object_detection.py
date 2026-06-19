#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import tf2_ros
import tf2_geometry_msgs # 用於轉換 PoseStamped
from geometry_msgs.msg import PoseStamped, PointStamped # 引入 PointStamped
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray
import torch
import threading # 引入 threading
import os # 引入 os 模組來處理路徑和檔案
import glob # << 新增：用於查找檔案
import json

class ErrorCompensationModel:
    def __init__(self, save_file=None):
        self.error_database = {}
        self.min_samples = 5
        self.max_samples = 50
        if save_file is None:
            # 使用腳本所在目錄，這樣更靈活
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_file = os.path.join(script_dir, "error_compensation_data.json")
        self.save_file = save_file
        
        # 程式啟動時載入之前的資料
        self.load_data()
    
    def save_data(self):
        """保存補償資料到檔案"""
        try:
            # 將numpy陣列轉換為列表以便JSON序列化
            save_data = {}
            for obj_class, errors in self.error_database.items():
                save_data[obj_class] = [error.tolist() for error in errors]
            
            with open(self.save_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            rospy.loginfo(f"Error compensation data saved to {self.save_file}")
        except Exception as e:
            rospy.logerr(f"Failed to save error compensation data: {e}")
    
    def load_data(self):
        """從檔案載入補償資料"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    save_data = json.load(f)
                
                # 將列表轉換回numpy陣列
                for obj_class, errors in save_data.items():
                    self.error_database[obj_class] = [np.array(error) for error in errors]
                
                rospy.loginfo(f"Loaded error compensation data from {self.save_file}")
                
                # 顯示載入的統計資訊
                for obj_class in self.error_database:
                    count = len(self.error_database[obj_class])
                    rospy.loginfo(f"  {obj_class}: {count} samples")
            else:
                rospy.loginfo(f"No existing compensation data found at {self.save_file}")
        except Exception as e:
            rospy.logerr(f"Failed to load error compensation data: {e}")
            self.error_database = {}
    
    def record_grab_result(self, predicted_coord, actual_coord, object_class):
        """記錄抓取結果用於學習"""
        if object_class not in self.error_database:
            self.error_database[object_class] = []
        
        error = np.array(actual_coord) - np.array(predicted_coord)
        self.error_database[object_class].append(error)
        
        # 保持最近50次的記錄
        if len(self.error_database[object_class]) > self.max_samples:
            self.error_database[object_class].pop(0)
        
        rospy.loginfo(f"Recorded error for {object_class}: [{error[0]:.3f}, {error[1]:.3f}, {error[2]:.3f}]")
        
        # 自動保存資料
        self.save_data()
    
    def get_compensation(self, predicted_coord, object_class):
        """取得補償後的座標"""
        if (object_class in self.error_database and 
            len(self.error_database[object_class]) >= self.min_samples):
            
            avg_error = np.mean(self.error_database[object_class], axis=0)
            compensated_coord = np.array(predicted_coord) + avg_error
            
            # 移除錯誤的 recording_mode 檢查
            rospy.loginfo_throttle(2.0, f"Applied compensation for {object_class}: "
                        f"error=[{avg_error[0]:.3f}, {avg_error[1]:.3f}, {avg_error[2]:.3f}]")
            
            return tuple(compensated_coord)
        
        return predicted_coord
    
    def get_statistics(self, object_class):
        """取得統計資訊"""
        if object_class in self.error_database:
            errors = np.array(self.error_database[object_class])
            return {
                'sample_count': len(errors),
                'mean_error': np.mean(errors, axis=0),
                'std_error': np.std(errors, axis=0),
                'max_error': np.max(np.abs(errors), axis=0)
            }
        return None

class MultiFrameCoordinateFilter:
    def __init__(self, window_size=10):
        self.coordinate_history = {}  # 按物件類別分別記錄
        self.window_size = window_size
        self.stability_threshold = 0.02  # 2cm穩定性閾值
    
    def add_coordinate(self, coord, object_class, recording_mode=False):
        """添加新座標並返回濾波後的座標"""
        if object_class not in self.coordinate_history:
            self.coordinate_history[object_class] = []
        
        self.coordinate_history[object_class].append(coord)
        if len(self.coordinate_history[object_class]) > self.window_size:
            self.coordinate_history[object_class].pop(0)
        
        # 使用加權平均，最新的座標權重更高
        if len(self.coordinate_history[object_class]) >= 3:
            weights = np.linspace(0.5, 1.0, len(self.coordinate_history[object_class]))
            coords_array = np.array(self.coordinate_history[object_class])
            filtered_coord = np.average(coords_array, axis=0, weights=weights)
            
            # 檢查穩定性（只在非記錄模式時輸出）
            if len(self.coordinate_history[object_class]) >= 5 and not recording_mode:
                recent_coords = coords_array[-5:]
                stability = np.std(recent_coords, axis=0)
                is_stable = np.all(stability < self.stability_threshold)
                
                rospy.loginfo_throttle(2.0, f"Coordinate stability for {object_class}: "
                            f"std=[{stability[0]:.4f}, {stability[1]:.4f}, {stability[2]:.4f}], "
                            f"stable={is_stable}")
                
                
            
            return tuple(filtered_coord)
        
        return coord
    
    def is_stable(self, object_class):
        """檢查座標是否穩定"""
        if (object_class in self.coordinate_history and 
            len(self.coordinate_history[object_class]) >= 5):
            
            coords_array = np.array(self.coordinate_history[object_class][-5:])
            stability = np.std(coords_array, axis=0)
            return np.all(stability < self.stability_threshold)
        
        return False
    
    def reset(self, object_class=None):
        """重置座標歷史"""
        if object_class:
            if object_class in self.coordinate_history:
                self.coordinate_history[object_class] = []
        else:
            self.coordinate_history = {}
            
class RGBD_ObjectDetector:
    def __init__(self):
        rospy.init_node('rgbd_object_detector', anonymous=True)
        self.bridge = CvBridge()
        self.camera_matrix = None
        
        # 用於儲存最新處理好的影像
        self.latest_image = None
        self.image_lock = threading.Lock() # 用於保護 latest_image 的線程安全

        # 初始化 TF2 Buffer 和 Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 創建一個發布器，用於發布轉換到 world 座標系後的點 (可選，方便驗證)
        self.world_point_pub = rospy.Publisher('/detected_object_world_point', PointStamped, queue_size=10)

        # --- 新增：誤差補償和多幀融合組件 ---
        self.error_compensation = ErrorCompensationModel()
        self.coordinate_filter = MultiFrameCoordinateFilter()
        self.grab_feedback_enabled = True  # 是否啟用抓取反饋學習
        # --- 結束：新增組件 ---

        # --- 新增：座標記錄相關變數 ---
        self.current_detected_objects = []  # 儲存當前檢測到的物件
        self.recording_mode = False  # 是否處於記錄模式
        self.pending_ground_truth = {}  # 等待輸入的ground truth座標
        # --- 結束：新增變數 ---

        # --- 新增：截圖相關設定 ---
        self.screenshot_path = "screenshots"
        if not os.path.exists(self.screenshot_path):
            os.makedirs(self.screenshot_path)
            print(f"截圖將保存在資料夾: {os.path.abspath(self.screenshot_path)}")
        
        self.screenshot_count = self.get_initial_screenshot_count()
        print(f"初始截圖計數器設定為: {self.screenshot_count}")
        # --- 結束：截圖相關設定 ---

         # --- 新增：錄影相關設定 ---
        self.video_path = "recordings"
        if not os.path.exists(self.video_path):
            os.makedirs(self.video_path)
            print(f"影片將保存在資料夾: {os.path.abspath(self.video_path)}")

        self.frames_path = os.path.join(self.video_path, "frames")
        if not os.path.exists(self.frames_path):  # 確保 frames_path 資料夾存在
            os.makedirs(self.frames_path)
            print(f"提取的幀將保存在資料夾: {os.path.abspath(self.frames_path)}")

        
        
        self.is_recording = False
        self.video_writer = None
        self.frame_count = 0
        self.frames_path = os.path.join(self.video_path, "frames")
        self.extract_frames = False  # 是否在錄影時同時提取幀
        self.current_frames_dir = None  # 初始化為 None
        self.latest_rgb_image = None  # 初始化 latest_rgb_image
        # --- 結束：錄影相關設定 ---

        self.init_model()

        if hasattr(self, 'model') and self.model is not None:
            self.color_sub = message_filters.Subscriber('/r200/camera/color/image_raw', Image)
            self.depth_sub = message_filters.Subscriber('/r200/camera/depth/image_raw', Image)
            self.camera_info_sub = rospy.Subscriber('/r200/camera/color/camera_info',
                                                    CameraInfo,
                                                    self.camera_info_callback)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.color_sub, self.depth_sub], 10, 0.1)
            self.ts.registerCallback(self.rgbd_callback)
            self.result_pub = rospy.Publisher('/object_detection_result', Image, queue_size=1) # 減少 queue_size
            self.marker_pub = rospy.Publisher('/object_markers', MarkerArray, queue_size=10)
            print("已啟動RGB-D物件偵測節點...")
            print(f"按 's' 鍵進行截圖, 截圖將保存在 '{self.screenshot_path}' 資料夾")
            print(f"按 'r' 鍵開始/停止錄影, 影片將保存在 '{self.video_path}' 資料夾")
            print(f"按 'f' 鍵切換是否在錄影時同時提取幀")
            print("按 'c': 記錄當前物件座標用於補償學習")
            print("按 'v': 查看補償統計資訊")
            print("按 'q': 退出")

        else:
            print("模型初始化失敗，無法啟動物件偵測節點")
            rospy.signal_shutdown("Model initialization failed") # 模型失敗則退出

    def record_current_coordinates_for_compensation(self):
        """記錄當前檢測到的物件座標用於補償學習"""
        if not self.current_detected_objects:
            rospy.logwarn("目前沒有檢測到任何物件，無法記錄座標")
            return
        
        rospy.loginfo("=== 開始記錄座標用於補償學習 ===")
        rospy.loginfo(f"檢測到 {len(self.current_detected_objects)} 個物件:")
        
        for i, obj in enumerate(self.current_detected_objects):
            obj_class = obj['class']
            predicted_coord = obj['position_3d']
            
            rospy.loginfo(f"{i+1}. {obj_class} - 預測座標: [{predicted_coord[0]:.3f}, {predicted_coord[1]:.3f}, {predicted_coord[2]:.3f}]")
        
        # 提示使用者輸入實際座標
        self.prompt_for_ground_truth()
    
    def prompt_for_ground_truth(self):
        """提示使用者輸入實際座標"""
        rospy.loginfo("請在終端中輸入實際的物件座標 (格式: x y z):")
        rospy.loginfo("例如: 1.0 0.15 0.775")
        rospy.loginfo("如果有多個物件，請依序輸入各物件的實際座標")
        
        # 設定標誌，讓主循環知道需要處理輸入
        self.recording_mode = True
        
        # 在新線程中處理輸入，避免阻塞主程式
        import threading
        input_thread = threading.Thread(target=self.handle_ground_truth_input)
        input_thread.daemon = True
        input_thread.start()
    
    def handle_ground_truth_input(self):
        """處理ground truth座標輸入"""
        try:
            # 清空終端並顯示明顯的提示
            import os
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("=" * 60)
            print("🎯 座標記錄模式 - 所有日誌已暫停")
            print("=" * 60)
            
            for i, obj in enumerate(self.current_detected_objects):
                obj_class = obj['class']
                # 使用世界座標系的預測座標
                if 'world_position_3d' in obj:
                    predicted_coord = obj['world_position_3d']
                else:
                    # 如果沒有世界座標，跳過或使用相機座標
                    print(f"⚠️ 物件 {obj_class} 沒有世界座標資訊，跳過")
                    continue
                
                print(f"\n📍 物件 {i+1}: {obj_class}")
                print(f"🔮 預測座標: [{predicted_coord[0]:.3f}, {predicted_coord[1]:.3f}, {predicted_coord[2]:.3f}]")
                print("-" * 40)
                
                while True:
                    try:
                        user_input = input("✏️  請輸入實際座標 (x y z) 或 's' 跳過: ").strip()
                        
                        if user_input.lower() == 's':
                            print(f"⏭️  跳過物件 {obj_class}")
                            break
                        
                        coords = user_input.split()
                        if len(coords) != 3:
                            print("❌ 格式錯誤，請輸入三個數字 (x y z)")
                            continue
                        
                        actual_coord = [float(coords[0]), float(coords[1]), float(coords[2])]
                        
                        # 記錄到補償模型
                        self.error_compensation.record_grab_result(
                            predicted_coord, actual_coord, obj_class
                        )
                        
                        print(f"✅ 已記錄 {obj_class} 的補償資料")
                        break
                        
                    except ValueError:
                        print("❌ 輸入格式錯誤，請輸入數字")
                    except Exception as e:
                        print(f"❌ 輸入處理錯誤: {e}")
            
            print("\n" + "=" * 60)
            print("🎉 座標記錄完成 - 恢復正常運行")
            print("=" * 60)
            self.recording_mode = False
            
        except Exception as e:
            print(f"處理ground truth輸入時出錯: {e}")
            self.recording_mode = False

    def show_compensation_statistics(self):
        """顯示補償統計資訊"""
        rospy.loginfo("=== 補償統計資訊 ===")
        
        if not self.error_compensation.error_database:
            rospy.loginfo("目前沒有補償資料")
            return
        
        for obj_class in self.error_compensation.error_database:
            stats = self.error_compensation.get_statistics(obj_class)
            if stats:
                rospy.loginfo(f"\n{obj_class}:")
                rospy.loginfo(f"  樣本數量: {stats['sample_count']}")
                rospy.loginfo(f"  平均誤差: [{stats['mean_error'][0]:.3f}, {stats['mean_error'][1]:.3f}, {stats['mean_error'][2]:.3f}]")
                rospy.loginfo(f"  標準差: [{stats['std_error'][0]:.3f}, {stats['std_error'][1]:.3f}, {stats['std_error'][2]:.3f}]")
                rospy.loginfo(f"  最大誤差: [{stats['max_error'][0]:.3f}, {stats['max_error'][1]:.3f}, {stats['max_error'][2]:.3f}]")

    def record_grab_feedback(self, predicted_coord, actual_coord, object_class, success=True):
        """記錄抓取反饋用於學習改進"""
        if self.grab_feedback_enabled and success:
            self.error_compensation.record_grab_result(predicted_coord, actual_coord, object_class)
            rospy.loginfo(f"Grab feedback recorded for {object_class}")

    def get_compensation_statistics(self):
        """取得所有物件的補償統計資訊"""
        stats = {}
        for obj_class in self.error_compensation.error_database:
            stats[obj_class] = self.error_compensation.get_statistics(obj_class)
        return stats

    def reset_learning_data(self, object_class=None):
        """重置學習數據"""
        if object_class:
            if object_class in self.error_compensation.error_database:
                self.error_compensation.error_database[object_class] = []
            self.coordinate_filter.reset(object_class)
        else:
            self.error_compensation.error_database = {}
            self.coordinate_filter.reset()
        
        rospy.loginfo(f"Learning data reset for {object_class if object_class else 'all objects'}")

    # --- 新增：獲取初始截圖計數的方法 ---
    def get_initial_screenshot_count(self):
        """檢查截圖資料夾中現有的檔案，以決定初始的截圖計數。"""
        highest_num = -1
        # 查找所有 screenshot_XXXX.png 格式的檔案
        existing_files = glob.glob(os.path.join(self.screenshot_path, "screenshot_*.png"))
        for f_path in existing_files:
            try:
                # 從檔名中提取數字部分
                filename = os.path.basename(f_path) # 例如 screenshot_0012.png
                num_str = filename.replace("screenshot_", "").replace(".png", "")
                num = int(num_str)
                if num > highest_num:
                    highest_num = num
            except ValueError:
                # 如果檔名格式不符，忽略
                pass
        return highest_num + 1 # 下一個計數從最大編號+1開始
    # --- 結束：獲取初始截圖計數的方法 ---

    def start_recording(self):
        """開始錄影"""
        if self.is_recording:
            rospy.logwarn("已經在錄影中")
            return
        
        # 獲取時間戳作為檔案名稱
        timestamp = rospy.Time.now().to_sec()
        video_filename = os.path.join(self.video_path, f"video_{int(timestamp)}.mp4")
        
        # 重置 current_frames_dir
        self.current_frames_dir = None

        # 如果需要提取幀，創建對應的資料夾
        if self.extract_frames:
            frames_dir = os.path.join(self.frames_path, f"video_{int(timestamp)}")
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            self.current_frames_dir = frames_dir
        
        # 獲取影像尺寸
        with self.image_lock:
            if self.latest_rgb_image is not None:
                height, width = self.latest_rgb_image.shape[:2]
            else:
                height, width = 480, 640  # 預設值
        
        # 初始化 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 編碼
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width, height))
        
        self.is_recording = True
        self.frame_count = 0
        rospy.loginfo(f"開始錄影: {video_filename}")

    def stop_recording(self):
        """停止錄影"""
        if not self.is_recording:
            rospy.logwarn("沒有進行中的錄影")
            return
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        rospy.loginfo(f"停止錄影，共記錄 {self.frame_count} 幀")

    def toggle_recording(self):
        """切換錄影狀態"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def toggle_frame_extraction(self):
        """切換是否在錄影時同時提取幀"""
        if self.is_recording:
            rospy.logwarn("錄影進行中，無法切換幀提取狀態。請先停止錄影。")
            return
            
        self.extract_frames = not self.extract_frames
        status = "開啟" if self.extract_frames else "關閉"
        rospy.loginfo(f"錄影時提取幀功能已{status}")

    def camera_info_callback(self, camera_info_msg):
        with self.image_lock: # 確保線程安全 (雖然此處影響不大，但養成習慣)
            if self.camera_matrix is None:
                self.camera_matrix = np.array(camera_info_msg.K).reshape(3, 3)
                print("Camera Metrix: ", camera_info_msg.K)
                print("已獲取相機參數")

    def init_model(self):
        try:
            # 載入您訓練好的模型，確保路徑正確
            best_pt_path = 'object_dataset/v5/runs/detect/yolo11_custom_training/weights/best.pt' # <--- 請確認這是正確的路徑
            self.model = YOLO(best_pt_path)
            self.confidence_threshold = 0.25
            print(f"已載入自訂訓練模型: {best_pt_path}")
            print("CUDA available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                print("GPU Name:", torch.cuda.get_device_name(0))
        except Exception as e:
            print(f"載入模型時出錯: {e}")
            self.model = None

    def rgbd_callback(self, rgb_msg, depth_msg):
        """僅處理影像和偵測，不顯示"""
        try:
            # 如果處於記錄模式，暫停處理以避免日誌干擾
            if self.recording_mode:
                return
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

            # 檢查相機矩陣是否已獲取
            local_camera_matrix = None
            with self.image_lock:
                if self.camera_matrix is None:
                    # rospy.logwarn_throttle(5, "Camera matrix not available yet.")
                    return # 如果沒有相機矩陣，暫時不處理
                local_camera_matrix = self.camera_matrix.copy()
                self.latest_rgb_image = rgb_image.copy() # << 新增：保存原始 RGB 影像

            # --- 錄影處理 (移到這裡) ---
            if self.is_recording and self.video_writer is not None:
                with self.image_lock:
                    if self.latest_rgb_image is not None:
                        # 寫入影片
                        self.video_writer.write(self.latest_rgb_image)
                        
                        # 如果需要同時提取幀
                        if self.extract_frames and self.current_frames_dir is not None:
                            try:
                                frame_filename = os.path.join(self.current_frames_dir, f"frame_{self.frame_count:06d}.jpg")
                                cv2.imwrite(frame_filename, self.latest_rgb_image)
                            except Exception as e:
                                rospy.logwarn(f"提取幀時出錯: {e}")
                        
                        self.frame_count += 1
            # --- 結束：錄影處理 ---

            # 執行物件偵測 (使用複製的相機矩陣)
            detected_objects = self.detect_objects(rgb_image, depth_image, local_camera_matrix)

            # --- 新增：更新當前檢測到的物件 ---
            self.current_detected_objects = detected_objects
            # --- 結束：更新 ---

             # --- 座標轉換與驗證 ---
            for obj in detected_objects:
                # 創建 PointStamped 消息，表示相機座標系中的點
                point_camera = PointStamped()
                # 重要：header.stamp 最好使用影像的時間戳，以便 tf 查找對應時間的轉換
                point_camera.header.stamp = rgb_msg.header.stamp # 使用 RGB 影像的時間戳
                # 重要：frame_id 必須是您計算 P_c 時所基於的座標系
                # 通常是光學座標系，例如 'camera_color_optical_frame'
                # 請根據您的相機 TF 設定確認
                point_camera.header.frame_id = "color" # <--- 請確認這個 Frame ID
                point_camera.point.x = obj['position_3d'][0]
                point_camera.point.y = obj['position_3d'][1]
                point_camera.point.z = obj['position_3d'][2]

                try:
                    # 使用 tf_buffer 將點從相機座標系轉換到世界座標系 ('world' 或 'map')
                    # 使用 rospy.Time(0) 或 rgb_msg.header.stamp 查找轉換
                    # rospy.Duration(1.0) 是等待轉換可用的超時時間
                    point_world = self.tf_buffer.transform(point_camera, 'world', rospy.Duration(1.0)) # <--- 目標座標系設為 'world'

                    # 將世界座標添加到物件資訊中
                    obj['world_position_3d'] = (point_world.point.x, point_world.point.y, point_world.point.z)

                    # 只在非記錄模式時輸出詳細日誌
                    if not self.recording_mode:
                        rospy.loginfo_throttle(2.0, f"Object '{obj['class']}' detected at camera frame ({point_camera.header.frame_id}): "
                                    f"[{point_camera.point.x:.3f}, {point_camera.point.y:.3f}, {point_camera.point.z:.3f}]")
                        rospy.loginfo_throttle(2.0, f"Transformed to world frame ('{point_world.header.frame_id}'): "
                                    f"[{point_world.point.x:.3f}, {point_world.point.y:.3f}, {point_world.point.z:.3f}]")

                    # (可選) 發布轉換後的點
                    self.world_point_pub.publish(point_world)

                    # --- 在這裡加入您的驗證邏輯 ---
                    # 例如，比較 point_world.point 的 x, y, z 與您在 Gazebo 中設定的 ground truth 座標
                    ground_truth_x = 1.0 # 您設定的物件世界 X 座標
                    ground_truth_y = 0.15 # 您設定的物件世界 Y 座標
                    ground_truth_z = 0.775 # 您設定的物件世界 Z 座標
                    tolerance = 0.05 # 設定允許的誤差範圍 (例如 5cm)

                    error_x = abs(point_world.point.x - ground_truth_x)
                    error_y = abs(point_world.point.y - ground_truth_y)
                    error_z = abs(point_world.point.z - ground_truth_z)

                    # 只在非記錄模式時輸出驗證結果
                    if not self.recording_mode:
                        if error_x < tolerance and error_y < tolerance and error_z < tolerance:
                            rospy.loginfo_throttle(2.0, f"Coordinates MATCH ground truth within tolerance {tolerance}m.")
                        else:
                            rospy.logwarn_throttle(2.0, f"Coordinates MISMATCH ground truth! Error (x,y,z): "
                                        f"[{error_x:.3f}, {error_y:.3f}, {error_z:.3f}]")
                    # ---------------------------------

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    # 錯誤訊息仍然顯示，但使用throttle避免過多輸出
                    if not self.recording_mode:
                        rospy.logwarn_throttle(5.0, f"TF transform error: {e}")

            # 準備視覺化結果影像
            result_image = self.visualize_results(rgb_image, detected_objects, local_camera_matrix)

            # 發布影像結果
            result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
            self.result_pub.publish(result_msg)

            # 發布 3D Markers
            markers = self.create_3d_markers(detected_objects, local_camera_matrix)
            self.marker_pub.publish(markers)

            # 更新最新影像以供主線程顯示
            with self.image_lock:
                self.latest_image = result_image.copy()

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in rgbd_callback: {e}")


    # 修改 detect_objects 和 create_3d_markers 以接收 camera_matrix 參數
    # def detect_objects(self, rgb_image, depth_image, camera_matrix):
    #     detected_objects = []
    #     if self.model is not None and camera_matrix is not None: # 增加 camera_matrix 檢查
    #         results = self.model(rgb_image, conf=self.confidence_threshold, device='cuda')
    #         for r in results:
    #             boxes = r.boxes
    #             for box in boxes:
    #                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    #                 conf = float(box.conf[0].cpu().numpy())
    #                 cls_id = int(box.cls[0].cpu().numpy())
    #                 cls_name = self.model.names[cls_id] # 使用 self.model.names 獲取類別名稱

    #                 y1 = max(0, y1)
    #                 y2 = min(depth_image.shape[0], y2)
    #                 x1 = max(0, x1)
    #                 x2 = min(depth_image.shape[1], x2)

    #                 obj_depth = depth_image[y1:y2, x1:x2]
    #                 valid_depths = obj_depth[obj_depth > 0]

    #                 if len(valid_depths) > 0:
    #                     avg_depth = np.median(valid_depths)
    #                     print("depth: ", avg_depth)
    #                     center_x = (x1 + x2) / 2
    #                     center_y = (y1 + y2) / 2

    #                     fx = camera_matrix[0, 0]
    #                     fy = camera_matrix[1, 1]
    #                     cx = camera_matrix[0, 2]
    #                     cy = camera_matrix[1, 2]

    #                     z = avg_depth / 1000.0
    #                     x = (center_x - cx) * z / fx
    #                     y = (center_y - cy) * z / fy

    #                     # 原始座標（暫時不使用x座標校正）
    #                     original_coord = (x, y, z)
                        
    #                     # 應用多幀濾波
    #                     filtered_coord = self.coordinate_filter.add_coordinate(original_coord, cls_name, self.recording_mode)
                        
    #                     # 應用誤差補償
    #                     final_coord = self.error_compensation.get_compensation(filtered_coord, cls_name)
                        
    #                     # 只在非記錄模式時輸出座標處理資訊
    #                     if not self.recording_mode:
    #                         rospy.loginfo(f"Coordinate processing for {cls_name}:")
    #                         rospy.loginfo(f"  Original: [{original_coord[0]:.3f}, {original_coord[1]:.3f}, {original_coord[2]:.3f}]")
    #                         rospy.loginfo(f"  Filtered: [{filtered_coord[0]:.3f}, {filtered_coord[1]:.3f}, {filtered_coord[2]:.3f}]")
    #                         rospy.loginfo(f"  Final: [{final_coord[0]:.3f}, {final_coord[1]:.3f}, {final_coord[2]:.3f}]")

    #                     obj_info = {
    #                         'class': cls_name, 
    #                         'confidence': conf, 
    #                         'bbox': (x1, y1, x2, y2),
    #                         'position_3d': final_coord,
    #                         'depth': avg_depth,
    #                         'original_position_3d': original_coord,
    #                         'filtered_position_3d': filtered_coord,
    #                         'is_stable': self.coordinate_filter.is_stable(cls_name)
    #                     }
    #                     detected_objects.append(obj_info)

    #     return detected_objects

    def detect_objects(self, rgb_image, depth_image, camera_matrix):
        """
        對 RGB 影像進行 YOLO 物件偵測，並結合深度影像計算每個物件的 3D 座標。

        Args:
            self: 物件自身的實例。
            rgb_image (np.array): 用於物件偵測的 RGB 影像。
            depth_image (np.array): 對應的深度影像，數值單位應為毫米 (mm)。
            camera_matrix (np.array): 3x3 的相機內參矩陣。

        Returns:
            list: 一個包含偵測到的物件資訊的列表，每個物件是一個字典。
        """
        detected_objects = []

        # 確保模型和相機內參都已就緒
        if self.model is None or camera_matrix is None:
            return detected_objects

        # 執行 YOLOv8 物件偵測
        results = self.model(rgb_image, conf=self.confidence_threshold, device='cuda', verbose=False)

        # 從相機內參矩陣中提取參數
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # 處理偵測結果
        for r in results:
            for box in r.boxes:
                # --- 1. 提取 Bounding Box 和元數據 ---
                # 將 PyTorch 張量轉換為 NumPy 陣列
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                rospy.loginfo_throttle(2.0, f"2D object center: ({(x1+x2)/2}, {(y1+y2)/2}")
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]

                # --- 2. 計算穩健的深度值 ---
                # 確保 BBox 座標不會超出影像邊界
                y1_c = max(0, y1)
                y2_c = min(depth_image.shape[0], y2)
                x1_c = max(0, x1)
                x2_c = min(depth_image.shape[1], x2)

                # 從深度圖中裁剪出物件的區域 (Region of Interest, ROI)
                depth_roi = depth_image[y1_c:y2_c, x1_c:x2_c]

                # 過濾掉無效的深度值 (通常為 0)
                valid_depths = depth_roi[depth_roi > 0]

                # 如果 ROI 內沒有有效的深度數據，則跳過此物件
                if len(valid_depths) == 0:
                    continue
                
                # 使用中位數來計算深度，這對異常值更穩健
                median_depth_mm = np.median(valid_depths)
                
                # depth offset
                median_depth_mm += 325

                # --- 3. 逆投影到相機座標系 ---
                # 將深度值從毫米 (mm) 轉換為米 (m)
                z = median_depth_mm / 1000.0
                
                # 使用 BBox 的中心點進行逆投影
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # height, weight offset
                center_x += 40
                center_y += 8

                # 應用逆投影公式
                x = (center_x - cx) * z / fx
                y = (center_y - cy) * z / fy

                # --- 4. 座標後處理與數據打包 ---
                original_coord = (x, y, z)
                
                # 應用多幀濾波以平滑結果
                filtered_coord = self.coordinate_filter.add_coordinate(original_coord, cls_name, self.recording_mode)
                
                # 應用學習到的誤差模型進行補償
                final_coord = self.error_compensation.get_compensation(filtered_coord, cls_name)
                
                # 在非記錄模式下，打印詳細的除錯日誌
                if not self.recording_mode:
                    rospy.loginfo_throttle(2.0, f"Coordinate processing for '{cls_name}':")
                    rospy.loginfo_throttle(2.0, f"  - Depth (median): {median_depth_mm:.1f} mm")
                    rospy.loginfo_throttle(2.0, f"  - Original 3D (camera frame): [{original_coord[0]:.3f}, {original_coord[1]:.3f}, {original_coord[2]:.3f}]")
                    rospy.loginfo_throttle(2.0, f"  - Filtered 3D (camera frame): [{filtered_coord[0]:.3f}, {filtered_coord[1]:.3f}, {filtered_coord[2]:.3f}]")
                    rospy.loginfo_throttle(2.0, f"  - Final 3D (camera frame):    [{final_coord[0]:.3f}, {final_coord[1]:.3f}, {final_coord[2]:.3f}]")

                # 將所有資訊打包成一個字典
                obj_info = {
                    'class': cls_name, 
                    'confidence': conf, 
                    'bbox': (x1, y1, x2, y2),
                    'position_3d': final_coord,  # 最終用於 TF 轉換的座標
                    'depth_mm': median_depth_mm,
                    'original_position_3d': original_coord,
                    'filtered_position_3d': filtered_coord,
                    'is_stable': self.coordinate_filter.is_stable(cls_name)
                }
                detected_objects.append(obj_info)

        return detected_objects

    def visualize_results(self, rgb_image, detected_objects, camera_matrix): # 添加 camera_matrix 參數 (雖然此處未使用，但保持一致)
        result_image = rgb_image.copy()
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cls = obj['class']
            conf = obj['confidence']
            x, y, z = obj['position_3d']
            # print("x, y, z: ", [x, y, z]) # 移除非必要打印

            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls}: {conf:.2f}" # 簡化標籤以避免覆蓋
            # --- 標註文字改進 ---
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size
            text_x = x1
            text_y = y1 - 5 if y1 - 5 > text_h else y1 + text_h + 5
            # 畫底色
            cv2.rectangle(result_image, (text_x, text_y - text_h - 2), (text_x + text_w, text_y + 2), (0, 255, 0), -1)
            # 畫文字 (黑色)
            cv2.putText(result_image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            # --- (可選) 添加3D座標文字 ---
            # coord_label = f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
            # cv2.putText(result_image, coord_label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return result_image
    
    # 在RGBD_ObjectDetector類別中添加
    def publish_single_object_coordinate(self, obj, header):
        """發布單個物件的世界座標"""
        try:
            # 創建相機座標系的點
            point_camera = PointStamped()
            point_camera.header = header
            point_camera.header.frame_id = "color"
            point_camera.point.x = obj['position_3d'][0]
            point_camera.point.y = obj['position_3d'][1]
            point_camera.point.z = obj['position_3d'][2]
            
            # 轉換到世界座標系
            point_world = self.tf_buffer.transform(point_camera, 'world', rospy.Duration(1.0))
            
            # 發布世界座標
            self.world_point_pub.publish(point_world)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish object coordinate: {e}")

    def create_3d_markers(self, detected_objects, camera_matrix): # 添加 camera_matrix 參數
        marker_array = MarkerArray()
        if camera_matrix is None: # 再次檢查
             return marker_array

        for i, obj in enumerate(detected_objects):
            marker = Marker()
            marker.header.frame_id = "color" # 確保這個 frame_id 在 RViz 中可用
            marker.header.stamp = rospy.Time.now()
            marker.ns = "objects"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            x, y, z = obj['position_3d']
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0

            x1, y1, x2, y2 = obj['bbox']
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]

            # 避免除以零
            if fx == 0 or fy == 0 or z <= 0:
                continue

            width_meters = abs(width_pixels * z / fx) # 使用 abs 避免負值
            height_meters = abs(height_pixels * z / fy)
            # 估計深度，可以基於類別或設為固定值
            depth_meters = max(0.1, min(width_meters, height_meters)) # 簡易估計

            marker.scale.x = max(0.01, width_meters) # 設置最小尺寸
            marker.scale.y = max(0.01, depth_meters)
            marker.scale.z = max(0.01, height_meters)

            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.7
            marker.lifetime = rospy.Duration(0.5) # 設置短暫生命週期，避免舊標記殘留
            marker_array.markers.append(marker)

             # --- (可選) 添加文字 Marker ---
            text_marker = Marker()
            text_marker.header.frame_id = "color"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "object_texts"
            text_marker.id = 1000 + i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + marker.scale.z / 2 + 0.05 # 在立方體上方
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.08
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0
            text_marker.lifetime = rospy.Duration(0.5)
            text_marker.text = f"{obj['class']} ({obj['confidence']:.2f})"
            marker_array.markers.append(text_marker)

        return marker_array

    # --- 新增：截圖方法 ---
    def save_screenshot(self):
        image_to_save = None
        with self.image_lock:
            if self.latest_rgb_image is not None:
                image_to_save = self.latest_rgb_image.copy()
        
        if image_to_save is not None:
            # 使用 self.screenshot_count 作為當前檔案的編號
            filename = os.path.join(self.screenshot_path, f"screenshot_{self.screenshot_count:04d}.png")
            try:
                cv2.imwrite(filename, image_to_save)
                rospy.loginfo(f"截圖已保存: {filename}")
                self.screenshot_count += 1 # 保存成功後才增加計數器
            except Exception as e:
                rospy.logerr(f"保存截圖失敗: {e}")
        else:
            rospy.logwarn("沒有可用的 RGB 影像進行截圖")
    # --- 結束：截圖方法 ---

    def run(self):
        """運行主循環，處理顯示"""
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            display_image = None
            with self.image_lock:
                if self.latest_image is not None:
                    display_image = self.latest_image.copy()

            if display_image is not None:
                # 顯示錄影和幀擷取狀態
                status_y = 30
                
                # 顯示錄影狀態
                if self.is_recording:
                    cv2.putText(display_image, "REC", (20, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_image, "Ready", (20, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 顯示記錄模式狀態
                if self.recording_mode:
                    cv2.putText(display_image, "RECORDING COORDS", (300, status_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 顯示幀擷取狀態
                extract_status = "EXTRACT: ON" if self.extract_frames else "EXTRACT: OFF"
                extract_color = (0, 255, 255) if self.extract_frames else (128, 128, 128)
                cv2.putText(display_image, extract_status, (120, status_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, extract_color, 2)
                
                # 在底部顯示操作提示
                h = display_image.shape[0]
                cv2.putText(display_image, "r: Record | f: Extract | s: Screenshot | c: Record Coords | v: Stats | q: Quit", 
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('RGB-D Object Detection', display_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 按 'q' 退出
                    if self.is_recording:
                        self.stop_recording()
                    # 程式結束前保存資料
                    self.error_compensation.save_data()
                    rospy.signal_shutdown("User requested exit.")
                    break
                elif key == ord('s'):  # 按 's' 截圖
                    self.save_screenshot()
                elif key == ord('r'):  # 按 'r' 開始/停止錄影
                    self.toggle_recording()
                elif key == ord('f'):  # 按 'f' 切換是否提取幀
                    if not self.is_recording:
                        self.toggle_frame_extraction()
                    else:
                        rospy.logwarn("錄影進行中，無法切換幀提取狀態。請先停止錄影。")
                elif key == ord('c'):  # 按 'c' 記錄當前座標
                    if not self.recording_mode:
                        self.record_current_coordinates_for_compensation()
                    else:
                        rospy.logwarn("正在記錄座標中，請完成當前記錄後再試")
                elif key == ord('v'):  # 按 'v' 查看統計資訊
                    self.show_compensation_statistics()

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS interrupt received, shutting down.")
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = RGBD_ObjectDetector()
        # 檢查模型是否成功初始化
        if hasattr(detector, 'model') and detector.model is not None:
             detector.run()
        else:
             print("Detector initialization failed, exiting.")
    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        cv2.destroyAllWindows() # 再次確保關閉視窗
