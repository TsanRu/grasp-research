import rospy
import moveit_commander
from moveit_commander.planning_scene_interface import PlanningSceneInterface
from moveit_commander.move_group import MoveGroupCommander
from moveit_commander import RobotCommander
from geometry_msgs.msg import Pose
import time
import numpy as np
from ur_control.arm import Arm
import actionlib
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
import copy
import geometry_msgs.msg
import rospkg
from std_msgs.msg import Header
from gazebo_ros_link_attacher.srv import Attach, AttachRequest, AttachResponse
from gazebo_msgs.srv import SetLinkProperties, SetLinkPropertiesRequest, GetLinkProperties
from std_srvs.srv import Empty
import tf2_ros
from gazebo_msgs.srv import GetModelState
from tf.transformations import quaternion_from_euler
import sys
import os
from gazebo_msgs.srv import SpawnModel
import threading
import functools
import queue
from geometry_msgs.msg import PointStamped


# 全域變數儲存物件位置
# object_pose = None
# left_arm = None
# right_arm = None
left_gripper = None
right_gripper = None
attach_srv = None
detach_srv = None
scene = None
arm_mover = None
object_pose_raw = None  # 永遠儲存最新原始座標
object_pose_lock = threading.Lock()  # 避免資料競爭
latest_coordinates = None
coordinates_received = None
is_detecting = True  # 偵測階段標誌
detection_lock = threading.Lock()

# Gazebo 與 MoveIt! 的 `base_link` 偏移量與旋轉角度
X_BASE_LINK = 0.11  # Gazebo 內 base_link 的 X 偏移
Y_BASE_LINK = 0.0   # Gazebo 內 base_link 的 Y 偏移
Z_BASE_LINK = 0.69  # Gazebo 內 base_link 的 Z 偏移
YAW = np.pi / 2     # base_link 在 Gazebo 內旋轉 +90° (radians)

class ArmMover:
    def __init__(self):
        # 佇列系統
        self.right_arm_queue = queue.Queue()
        self.left_arm_queue = queue.Queue()
        
        # MoveGroup 實例
        self.right_arm = moveit_commander.MoveGroupCommander('rightarm')
        self.left_arm = moveit_commander.MoveGroupCommander('leftarm')
        
        # 執行緒安全機制
        self.right_lock = threading.Lock()
        self.left_lock = threading.Lock()
        
        # 狀態追蹤
        self.right_busy = False
        self.left_busy = False

        # 工作執行緒
        self.right_worker = threading.Thread(
            target=self._arm_worker, 
            args=(self.right_arm, self.right_arm_queue, self.right_lock),
            daemon=True
        )
        self.left_worker = threading.Thread(
            target=self._arm_worker,
            args=(self.left_arm, self.left_arm_queue, self.left_lock),
            daemon=True
        )

    def start_threads(self):
        self.right_worker.start()
        self.left_worker.start()

    def _arm_worker(self, arm, q, lock):
        """通用手臂工作執行緒"""
        while not rospy.is_shutdown():
            try:
                target = q.get(block=True, timeout=0.1)
                self._execute_single_target(arm, target, lock)
                q.task_done()
            except queue.Empty:
                continue

    def _right_arm_worker(self):
        self._arm_worker(self.right_arm, self.right_arm_queue, self.right_lock, True)

    def _left_arm_worker(self):
        self._arm_worker(self.left_arm, self.left_arm_queue, self.left_lock, False)

    def _execute_single_target(self, arm, target, is_right):
        """執行單個目標點"""
        try:
            # 建立目標姿勢
            target_pose = Pose()
            target_pose.position.x = target[0]
            target_pose.position.y = target[1]
            target_pose.position.z = target[2]
            target_pose.orientation.x = 1.0  # 固定方向

            # 規劃與執行
            (plan, fraction) = arm.compute_cartesian_path([target_pose], 0.01)
            if fraction > 0.95:
                arm.set_goal_position_tolerance(0.05)  # 設定夾取範圍為 5cm
                arm.set_goal_orientation_tolerance(0.05)  # 設定姿勢容錯為 0.05弧度
                arm.execute(plan, wait=True)  # 同步執行確保順序
                side = "RIGHT" if is_right else "LEFT"
                rospy.loginfo(f"[{side}] Moved to {target}")
            else:
                rospy.logwarn("Cartesian path planning failed")
        except Exception as e:
            rospy.logerr(f"Arm movement error: {str(e)}")

    def set_right_arm_target(self, target):
        print("inside set rightarm target function")
        """添加右手臂目標到佇列"""
        self.right_arm_queue.put(target)
        rospy.loginfo(f"Added right arm target: {target}")
        rospy.loginfo(f"Queue still remains: {self.right_arm_queue.qsize()}")

    def set_left_arm_target(self, target):
        print("inside set leftarm target function")
        """添加左手臂目標到佇列"""
        self.left_arm_queue.put(target)
        rospy.loginfo(f"Added left arm target: {target}")
        rospy.loginfo(f"Queue still remains: {self.right_arm_queue.qsize()}")

    def wait_for_right_arm_done(self):
        """等待右手臂任務完成"""
        self.right_arm_queue.join()
        rospy.loginfo("Right arm queue cleared")

    def wait_for_left_arm_done(self):
        """等待左手臂任務完成"""
        self.left_arm_queue.join()
        rospy.loginfo("Left arm queue cleared")

    def emergency_stop(self):
        """緊急停止所有手臂"""
        self.right_arm.stop()
        self.left_arm.stop()
        rospy.logwarn("Emergency stop activated")

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

def get_object_pose_in_moveit(object_name):
    """
    從 /gazebo/model_states 取得指定 cube 的座標，
    並轉換到 MoveIt! 的 `base_link` 座標。
    若找不到則回傳 None。
    """
    # 等待 Gazebo 傳送物件座標
    model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)

    # 找到 cube 在 Gazebo 中的索引
    try:
        idx = model_states.name.index(object_name)
    except ValueError:
        rospy.logerr(f"{object_name} not found in /gazebo/model_states")
        return None

    # 取得該 cube 在 Gazebo 世界座標的 pose
    cube_pose = model_states.pose[idx]
    x_gaz = cube_pose.position.x
    y_gaz = cube_pose.position.y
    z_gaz = cube_pose.position.z

    # 轉換到 MoveIt! `base_link` 座標
    return transform_world_to_moveit(x_gaz, y_gaz, z_gaz)

def initialize_moveit():

    global left_arm, right_arm, scene

    # 初始化 MoveIt 和 ROS 節點
    moveit_commander.roscpp_initialize([])
    rospy.init_node('dual_ur3_moveit_control', anonymous=True)

    # robot = moveit_commander.RobotCommander()
    # scene = moveit_commander.PlanningSceneInterface()
    # 設定要控制的手臂 (這裡設定為 'leftarm' 和 'rightarm')
    left_arm = MoveGroupCommander('leftarm')
    left_arm.set_planning_time(5)
    right_arm = MoveGroupCommander('rightarm')
    right_arm.set_planning_time(5)

    # add_object()
    # 設定規劃時所使用的參數
    # left_arm_group.set_planner_id("RRTConnectkConfigDefault")
    # right_arm_group.set_planner_id("RRTConnectkConfigDefault")

def initialize_gripper():

    global left_gripper, right_gripper

    # 連接到手臂夾爪的 action server
    left_gripper = actionlib.SimpleActionClient('/leftarm/gripper_controller/gripper_cmd', GripperCommandAction)
    right_gripper = actionlib.SimpleActionClient('/rightarm/gripper_controller/gripper_cmd', GripperCommandAction)

    # 等待 server 啟動
    left_gripper.wait_for_server()
    right_gripper.wait_for_server()

def initialize_link_attacher():

    global attach_srv, detach_srv
    attach_srv = rospy.ServiceProxy('/link_attacher_node/attach',Attach)
    attach_srv.wait_for_service()
    detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
    detach_srv.wait_for_service()

def move_arm_to_position(move_group, position):

    target_pose = Pose()
   
    # 這裡我們將夾爪的朝向設置為朝下，假設夾爪應該朝 Z 軸負方向
    # roll, pitch, yaw = 3.14159, 0.0, 0.0  # 根據需求調整這些角度
    # quat = quaternion_from_euler(roll, pitch, yaw)

    # target_pose.orientation.x = quat[0]
    # target_pose.orientation.y = quat[1]
    # target_pose.orientation.z = quat[2]
    # target_pose.orientation.w = quat[3]

    # 創建目標姿勢
    target_pose.position.x = position[0]
    target_pose.position.y = position[1]
    target_pose.position.z = position[2]
    # target_pose.orientation.w = 1.0  # 如果有需要，也可以設定其他旋轉參數
    target_pose.orientation.x = 1
    target_pose.orientation.y = 0
    target_pose.orientation.z = 0
    target_pose.orientation.w = 0

    waypoints = []
    waypoints.append(copy.deepcopy(target_pose))

    # 將新目標點加入 waypoints
    # waypoints.append(target_pose)

    # 計算 Cartesian Path
    (plan, fraction) = move_group.compute_cartesian_path(
        waypoints,   # 目標點列表
        0.01,        # 步長（resolution），影響軌跡平滑度（0.01 = 1cm）
    )

     # 執行規劃好的路徑
    if fraction > 0.95:  # 確保計算成功（路徑覆蓋率超過 95%）
        move_group.set_goal_position_tolerance(0.05)  # 設定夾取範圍為 5cm
        move_group.set_goal_orientation_tolerance(0.05)  # 設定姿勢容錯為 0.05弧度
        move_group.execute(plan, wait=True)
        rospy.loginfo(f"Successfully moved to {position} using Cartesian Path")
    else:
        rospy.logwarn("Failed to compute a valid Cartesian Path.")

    # # 設定目標姿勢
    # move_group.set_pose_target(target_pose)

    # move_group.set_planner_id("PRM")

    # # 規劃並執行運動
    # success = move_group.go(wait=True)
    # if success:
    #     rospy.loginfo(f"Successfully moved to {position}")
    # else:
    #     rospy.logwarn("Failed to move arm to the target position.")

    # 停止運動
    move_group.stop()
    # move_group.clear_pose_targets()

# 定義夾爪動作函數
def control_gripper(gripper_client, position, max_effort=1000.0):
    """
    控制夾爪的開合

    :param position: 夾爪目標位置 (0.0 = 打開, 0.08 = 關閉)
    :param max_effort: 夾爪最大施力
    """
    goal = GripperCommandGoal()
    goal.command.position = position  # 設定夾爪開合位置
    goal.command.max_effort = max_effort  # 設定夾爪最大夾持力
    gripper_client.send_goal(goal)
    # gripper_client.wait_for_result()

def wait_for_state_update(box_name, scene, box_is_known=False, box_is_attached=False, timeout=4):

    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
        # Test if the box is in attached objects
        attached_objects = scene.get_attached_objects([box_name])
        is_attached = len(attached_objects.keys()) > 0

        # Test if the box is in the scene.
        # Note that attaching the box will remove it from known_objects
        is_known = box_name in scene.get_known_object_names()

        # Test if we are in the expected state
        if (box_is_attached == is_attached) and (box_is_known == is_known):
            return True

        # Sleep so that we give other threads time on the processor
        rospy.sleep(0.1)
        seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False

# 這段代碼將物體（例如 hammer）添加到場景中並設置其位置
def add_object_to_scene(scene):

    # 初始化 rospkg
    rospack = rospkg.RosPack()

    # 假設模型存放在名為 'your_package' 的包內
    package_path = rospack.get_path('ur_gripper_gazebo')

    # 假設 'hammer.stl' 是你的鎚子模型檔案
    hammer_mesh_path = package_path + "/models/hammer/meshes/hammer.dae"

    box_name = "hammer"
    # 取得 cube 在 MoveIt! `base_link` 座標系下的位置
    cube_pose_moveit = get_object_pose_in_moveit(box_name)
    if cube_pose_moveit is None:
        rospy.logwarn(f"無法找到 {box_name} 位置，結束此流程。")
        return
    
    # 展開座標
    cube_x, cube_y, cube_z = cube_pose_moveit
    cube_z += 0.18  # 夾取高度補償
    print("cube: ", cube_x, cube_y, cube_z)

    # 設定模型的位置 (例如 X, Y, Z 座標)
    hammer_pose = geometry_msgs.msg.PoseStamped()
    hammer_pose.header.frame_id = "world"
    hammer_pose.pose.position.x = cube_x
    hammer_pose.pose.position.y = cube_y
    hammer_pose.pose.position.z = cube_z
    # hammer_pose.pose.position.x = 1
    # hammer_pose.pose.position.y = 1
    # hammer_pose.pose.position.z = 1
    hammer_pose.pose.orientation.w = 1.0  # 默認的四元數
    
    # 添加模型到場景
    scene.add_mesh("hammer", hammer_pose, hammer_mesh_path)

def get_link_properties(link_name):
    rospy.wait_for_service('/gazebo/get_link_properties')
    try:
        get_link_properties_srv = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        resp = get_link_properties_srv(link_name)
        if resp.success:
            return resp
        else:
            rospy.logwarn(f"無法獲取 {link_name} 的屬性: {resp.status_message}")
            return None
    except rospy.ServiceException as e:
        rospy.logerr(f"獲取屬性失敗: {e}")
        return None

def set_gravity(link_name, gravity_enabled):
    rospy.wait_for_service('/gazebo/set_link_properties')
    try:
        set_link_properties_srv = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)
        
        # 先讀取當前屬性
        link_props = get_link_properties(link_name)
        if link_props is None:
            return
        
        # 設定請求
        req = SetLinkPropertiesRequest()
        req.link_name = link_name
        req.gravity_mode = gravity_enabled  # 只變更重力屬性
        req.mass = link_props.mass  # 保持原始質量
        req.ixx = link_props.ixx  # 保持原始慣性矩陣
        req.iyy = link_props.iyy
        req.izz = link_props.izz
        req.ixy = link_props.ixy
        req.ixz = link_props.ixz
        req.iyz = link_props.iyz

        # 送出變更請求
        resp = set_link_properties_srv(req)
        if resp.success:
            rospy.loginfo(f"成功設定 {link_name} 的重力為 {gravity_enabled}")
        else:
            rospy.logwarn(f"設定失敗: {resp.status_message}")

    except rospy.ServiceException as e:
        rospy.logerr(f"服務呼叫失敗: {e}")

def call_service(service_name):
    """ 通用函數來呼叫 ROS 服務 """
    rospy.wait_for_service(service_name)
    try:
        service = rospy.ServiceProxy(service_name, Empty)
        service()
        rospy.loginfo(f"成功呼叫 {service_name}")
    except rospy.ServiceException as e:
        rospy.logerr(f"呼叫 {service_name} 失敗: {e}")

def get_object_pose_from_gazebo(model_name):
    """從 Gazebo 中獲取物體的位置和姿態"""
    rospy.wait_for_service('/gazebo/get_model_state')
    try:
        get_model_state_service = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_state = get_model_state_service(model_name, "")  # 空字符串代表世界座標系
        return model_state.pose.position, model_state.pose.orientation
    except rospy.ServiceException as e:
        rospy.logerr(f"無法獲取物體的狀態: {e}")
        return None, None

def update_object_pose(model_name, robot_name):

    """從 Gazebo 獲取物體的位置，並更新 TF 位置"""
    position, orientation = get_object_pose_from_gazebo(model_name)

    if position is None or orientation is None:
        rospy.logerr("無法獲取物體的狀態，無法更新位置")
        return

    # 創建 tf 廣播器
    tf_broadcaster = tf2_ros.TransformBroadcaster()
    transform = geometry_msgs.msg.TransformStamped()

    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = robot_name  # 物件將被 attach 的機械手座標系
    transform.child_frame_id = model_name  # 物體的名稱

    # 設定物體的位置和姿態
    transform.transform.translation.x = position.x
    transform.transform.translation.y = position.y
    transform.transform.translation.z = position.z

    transform.transform.rotation = orientation

    rospy.loginfo("從 Gazebo 更新物件的 TF 位置")
    tf_broadcaster.sendTransform(transform)
 
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

def object_pose_callback(msg):
    """接收 /object_pose 資訊並更新全域變數"""
    global object_pose_raw

    with object_pose_lock:
        object_pose_raw = msg.pose # 更新最新的物件位置

def get_current_pose_with_offset(x_offset=0, y_offset=0, z_offset=0):
    """即時取得當前座標並計算補償值"""
    global object_pose_raw, object_pose_lock
    print("current object pose raw position: ", object_pose_raw)
    
    with object_pose_lock:
        if object_pose_raw is None:
            return None
        
        z = object_pose_raw.position.z + 0.17 # 固定夾爪高度補償
        # 動態追加補償
        x = object_pose_raw.position.x + x_offset
        y = object_pose_raw.position.y + y_offset
        z += z_offset
        
    return (x, y, z)  # 直接返回數值而非物件

def coordinate_callback(msg):
    """簡化的全域回調函數"""
    global latest_coordinates, coordinates_received, is_detecting
    
    # 只有在偵測階段才更新座標
    with detection_lock:
        if is_detecting:
            latest_coordinates = (msg.point.x, msg.point.y, msg.point.z)
            coordinates_received = True
            print(f"🔍 Detecting: [{msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f}]")

def get_current_pose_with_offset_detected_object(x_offset=0, y_offset=0, z_offset=0):
    """簡化的座標取得函數"""
    global latest_coordinates, coordinates_received, is_detecting
    
    # 訂閱座標
    coord_sub = rospy.Subscriber('/detected_object_world_point', 
                               PointStamped, 
                               coordinate_callback)
    
    # 確保處於偵測階段
    with detection_lock:
        if not is_detecting:
            rospy.logwarn("⚠️ Not in detection phase, cannot get coordinates")
            return None
    
    # 等待接收座標
    rate = rospy.Rate(10)
    timeout = 10.0
    start_time = rospy.Time.now()
    
    rospy.loginfo("🔍 Waiting for coordinates in detection phase...")
    
    while not rospy.is_shutdown():
        if coordinates_received and latest_coordinates:
            # 鎖定座標，進入執行階段
            with detection_lock:
                is_detecting = False
                rospy.loginfo("🔒 Coordinates locked - entering execution phase")
            
            x, y, z = latest_coordinates
            z += 0.1
            adjusted_x = x + x_offset
            adjusted_y = y + y_offset
            adjusted_z = z + z_offset
            
            rospy.loginfo(f"📍 Locked coordinates: [{adjusted_x:.3f}, {adjusted_y:.3f}, {adjusted_z:.3f}]")
            return adjusted_x, adjusted_y, adjusted_z
        
        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logwarn("⚠️ Timeout waiting for coordinates")
            return None
            
        rate.sleep()
    
    return None

def start_detection():
    """開始偵測階段"""
    global is_detecting, coordinates_received
    with detection_lock:
        is_detecting = True
        coordinates_received = False
        rospy.loginfo("🔍 Starting detection phase")

def stop_detection():
    """停止偵測階段（進入執行階段）"""
    global is_detecting
    with detection_lock:
        is_detecting = False
        rospy.loginfo("🤖 Entering execution phase")

def is_in_detection_phase():
    """檢查是否處於偵測階段"""
    with detection_lock:
        return is_detecting

def reset_to_detection():
    """重置回偵測階段"""
    global is_detecting, coordinates_received, latest_coordinates
    with detection_lock:
        is_detecting = True
        coordinates_received = False
        latest_coordinates = None
        rospy.loginfo("🔄 Reset to detection phase")



def is_object_in_handover_area(object_name):
    """接收 /object_pose 資訊並更新全域變數"""
    global object_pose_raw, right_arm, left_arm

    GRASP_X_MIN = 0.4
    GRASP_X_MAX = 1.15
    GRASP_Y_MIN = -0.2
    GRASP_Y_MAX = 0.2
    GRASP_Z_MIN = 0.69
    GRASP_Z_MAX = 1.1

    # 訂閱物件位置
    subscribe_object_pose()

    # 等待物件位置數據
    rospy.loginfo("Waiting for object pose...")
    while object_pose_raw is None and not rospy.is_shutdown():
        rospy.sleep(0.1)  # 等待數據更新

    rospy.loginfo(f"Using object position: \n{object_pose_raw.position}")

    # object_pose = msg.pose  # 更新最新的物件位置
    # object_pose.position.z += 0.17

    # x = msg.pose.position.x
    # y = msg.pose.position.y
    # z = msg.pose.position.z

    # 判斷物件是否在可夾取範圍內
    if (GRASP_X_MIN <= object_pose_raw.position.x <= GRASP_X_MAX and 
        GRASP_Y_MIN <= object_pose_raw.position.y <= GRASP_Y_MAX and 
        GRASP_Z_MIN <= object_pose_raw.position.z <= GRASP_Z_MAX):

        rospy.loginfo("物件進入夾取範圍，執行夾取動作！")

        try:

            auto_attach(object_name)


        except rospy.ServiceException as e:

            rospy.logerr(f"執行夾取失敗: {e}")
      
def subscribe_object_pose():
    """訂閱 /object_pose topic"""
    rospy.Subscriber("/object_pose", PoseStamped, object_pose_callback)
    rospy.loginfo("Subscribed to /object_pose")

def move_hammer(right_arm, right_arm_gripper):

    # 控制夾爪(張開)
    control_gripper(right_arm_gripper, 0)
    rospy.sleep(1)

    # 設定目標位置 (x, y, z)
    right_target_position = [0.5922, 0.31, 1.0] 
    move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至物件上方

    right_target_position = [0.5922, 0.31, 0.96]
    move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至物件抓取位置

    # # 控制夾爪(夾取)
    # control_gripper(right_arm_gripper, 0.0129)
    # rospy.sleep(1)

    # right_target_position = [0.5922, 0.31, 1.0]
    # move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至物件抓取位置

def attach_hammer(left_arm, left_arm_gripper, right_arm, right_arm_gripper, attach_srv, detach_srv):

    # hammer_link = "hammer::link"

    req_right = AttachRequest()
    req_right.model_name_1 = "hammer"
    req_right.link_name_1 = "link"
    req_right.model_name_2 = "robot"    
    req_right.link_name_2 = "rightarm_wrist_3_link"

    req_left = AttachRequest()
    req_left.model_name_1 = "hammer"
    req_left.link_name_1 = "link"
    req_left.model_name_2 = "robot"
    req_left.link_name_2 = "leftarm_wrist_3_link"
   
    # 設定目標位置 (x, y, z)
    right_target_position = [0.630, 0.088, 0.846] 
    move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至物件上方

    # right_target_position = [0.7, 0.15, 0.87]
    # move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至物件抓取位置

    # link_props = get_link_properties(hammer_link)
    # rospy.loginfo(f"物件 {hammer_link} 的重力模式: {link_props.gravity_mode}")
    # set_gravity(hammer_link, False)

    # attach_srv.call(req_right)
    # control_gripper(right_arm_gripper, 0.01)
    
    # link_props = get_link_properties(hammer_link)
    # rospy.loginfo(f"物件 {hammer_link} 的重力模式: {link_props.gravity_mode}")

    # right_target_position = [0.7, 0.15, 1.0]
    # right_target_position = [0.5922, 0.31, 1.1]
    # move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至抓取位置上方

    # right_target_position = [0.7, 0.1, 1.0]
    # move_arm_to_position(right_arm, right_target_position)  # 移動右手臂至抓取位置前方

    # left_arm_position = [0.725, -0.10, 1.1]
    # move_arm_to_position(left_arm, left_arm_position)

    # left_arm_position = [0.725, -0.10, 1.0]
    # move_arm_to_position(left_arm, left_arm_position)

    # call_service('/gazebo/pause_physics')
    
    # print("everything stop")

    

    # rospy.sleep(1)
    # print("already sleep")

    # control_gripper(right_arm_gripper, 0)
    # detach_srv.call(req_right)
    # print("right arm has detached")
    # rospy.sleep(1)

    # update_object_pose("hammer", "leftarm_wrist_3_link")

    # attach_srv.call(req_left)
    # print("left arm has attached")
    # control_gripper(left_arm_gripper, 0)
    # rospy.sleep(1)

    # call_service('/gazebo/unpause_physics')

    # left_arm_position = [0.725, -0.10, 0.9]
    # move_arm_to_position(left_arm, left_arm_position)

    # print("left arm move")
    
    
    # set_gravity(hammer_link, True)

def grasping_function(x_offset=0, y_offset=0, z_offset=0, arm=None):
    """你的抓取流程函數"""
    try:
        
        # 1. 確保處於偵測階段
        if not is_in_detection_phase():
            rospy.loginfo("Not in detection phase, resetting...")
            reset_to_detection()
        
        # 2. 取得座標（會自動鎖定並進入執行階段）
        target_pose = get_current_pose_with_offset_detected_object(x_offset, y_offset, z_offset)
        
        if target_pose is None:
            rospy.logerr("❌ Failed to get coordinates")
            reset_to_detection()  # 失敗時重置
            return False
        
        x, y, z = target_pose
        rospy.loginfo(f"🎯 Executing grab at: [{x:.3f}, {y:.3f}, {z:.3f}]")
        
        # 3. 執行抓取（此時不會收到新座標）
        # 你的抓取程式碼...
        # move_arm_to_position(x, y, z)
        # close_gripper()
        move_arm_to_position(arm, target_pose)
        
        rospy.loginfo("✅ Grab completed")
        return True
        
    except Exception as e:
        rospy.logerr(f"❌ Grab failed: {e}")
        return False
    
    finally:
        # 4. 完成後重置回偵測階段
        reset_to_detection()

# 檢查當前狀態
def check_status():
    if is_in_detection_phase():
        print("🔍 Currently in DETECTION phase")
    else:
        print("🤖 Currently in EXECUTION phase")

def auto_attach(object_name):

    # 訂閱物件位置
    subscribe_object_pose()

    """執行 handover，使用訂閱到的物件位置"""
    global object_pose_raw, arm_mover

    # 等待物件位置數據
    rospy.loginfo("Waiting for object pose...")
    while object_pose_raw is None and not rospy.is_shutdown():
        rospy.sleep(0.1)  # 等待數據更新

    rospy.loginfo(f"Using object position: \n{object_pose_raw.position}")
    

    req_right = AttachRequest()
    req_right.model_name_1 = object_name
    req_right.link_name_1 = "link"
    req_right.model_name_2 = "robot"    
    req_right.link_name_2 = "rightarm_wrist_3_link"

    req_left = AttachRequest()
    req_left.model_name_1 = object_name
    req_left.link_name_1 = "link"
    req_left.model_name_2 = "robot"
    req_left.link_name_2 = "leftarm_wrist_3_link"

    # target_pos = get_current_pose_with_offset_detected_object(z_offset = 0.13)
    grasping_function(0, 0, 0.13, right_arm)
    # right_target_position = [x, y, z + 0.13]
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to upper object position


    # right_target_position = [x, y, z] 
    # target_pos = get_current_pose_with_offset_detected_object()
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to object position
    grasping_function(0, 0, 0, right_arm)


    # arm_mover.wait_for_right_arm_done()
    attach_srv.call(req_right)
    # control_gripper(right_arm_gripper, 0.01)
    # rospy.sleep(1)

    # right_target_position = [x, y, z + 0.13] 

    # target_pos = get_current_pose_with_offset_detected_object(z_offset = 0.13)
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to upper object position
    grasping_function(0, 0, 0.13, right_arm)
    # arm_mover.set_right_arm_target(target_pos)

    # target_pos = get_current_pose_with_offset_detected_object(x_offset = -0.28, y_offset= -0.10)
    # right_target_position = [x - 0.28, y -0.05, z + 0.13]
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to handover position
    grasping_function(-0.28, -0.1, 0, right_arm)


# ===================================================================================================================================================

    # target_pos = get_current_pose_with_offset_detected_object(x_offset=0.023, y_offset= -0.1, z_offset=0.1)
    # move_arm_to_position(left_arm, target_pos)  # move leftarm to upper handover position
    grasping_function(0.023, -0.1, 0.1, left_arm)

    # target_pos = get_current_pose_with_offset_detected_object(x_offset=0.023, y_offset= -0.1)
    # move_arm_to_position(left_arm, target_pos)  # move leftarm to handover position
    grasping_function(0.023, -0.1, 0, left_arm)

    # # # control_gripper(right_arm_gripper, 0)
    # arm_mover.wait_for_left_arm_done()
    # arm_mover.wait_for_right_arm_done()
    detach_srv.call(req_right)
    attach_srv.call(req_left)
    # # # control_gripper(left_arm_gripper, 0)

    # target_pos = get_current_pose_with_offset_detected_object(z_offset= 0.1)
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to avoid collision with leftarm
    grasping_function(0, 0, 0.1, left_arm)

    # target_pos = get_current_pose_with_offset_detected_object(x_offset=0.28, y_offset=0.1, z_offset= 0.1)
    # move_arm_to_position(right_arm, target_pos)  # move rightarm to avoid collision with leftarm
    grasping_function(0.28, 0.1, 0.1, left_arm)

    target_pos = get_current_pose_with_offset_detected_object(x_offset=-0.143, y_offset= -0.1)
    move_arm_to_position(left_arm, target_pos)  # move leftarm to upper handover position
    grasping_function(-0.143, -0.1, 0, left_arm)

    # # # left_target_position = [object_pose.position.x - 0.195, object_pose.position.y - 0.15, object_pose.position.z]
    # # # print("current position: ", [object_pose.position.x, object_pose.position.y, object_pose.position.z])
    # # # move_arm_to_position(left_arm, left_target_position)  # move leftarm to upper put-down position
    # arm_mover.set_left_arm_target([x - 0.195, y - 0.15, z])

    # # # left_target_position = [object_pose.position.x, object_pose.position.y - 0.15, 0.87]
    # # # print("current position: ", [object_pose.position.x, object_pose.position.y, object_pose.position.z])
    # # # move_arm_to_position(left_arm, left_target_position)  # move leftarm to put-down position
    # arm_mover.set_left_arm_target([x, y - 0.15, 0.87])

    # arm_mover.wait_for_left_arm_done()
    # arm_mover.wait_for_right_arm_done()
    # # # # control_gripper(left_arm_gripper, 0)
    # detach_srv.call(req_left)

    # # # left_target_position = [object_pose.position.x, object_pose.position.y - 0.15, object_pose.position.z + 0.13]
    # # # move_arm_to_position(left_arm, left_target_position)  # move leftarm to upper put-down position
    # arm_mover.set_left_arm_target([x, y - 0.15, z + 0.13])

    # arm_mover.wait_for_left_arm_done()
    # detach_srv.call(req_right)

def vision_based_handover_trigger(object_name):
    """
    【新函數】
    持續監聽視覺偵測結果，一旦偵測到的物件進入指定範圍，
    就鎖定該座標並觸發一次完整的抓取與交接序列。
    """
    global latest_coordinates, coordinates_received, is_detecting
    global right_arm, left_arm # 確保能取用到手臂的實例

    GRASP_X_MIN = 0.4
    GRASP_X_MAX = 1.15
    GRASP_Y_MIN = -0.2
    GRASP_Y_MAX = 0.2
    GRASP_Z_MIN = 0.69
    GRASP_Z_MAX = 1.1

    # 確保系統處於偵測模式，並開始訂閱視覺偵測的 Topic
    reset_to_detection()
    rospy.Subscriber('/detected_object_world_point', PointStamped, coordinate_callback)
    
    rospy.loginfo("--- 🤖 等待視覺偵測到物件進入 handover 區域 ---")
    
    rate = rospy.Rate(5) # 每秒檢查 5 次
    while not rospy.is_shutdown():
        # 創建一個局部變數來安全地讀取座標
        current_pose = None
        
        # 檢查是否收到了新的視覺座標
        with detection_lock:
            if coordinates_received and latest_coordinates:
                # 複製座標，避免在檢查時被覆寫
                current_pose = latest_coordinates
                # 重置標誌位，等待下一次新的偵測結果
                coordinates_received = False
        
        # 如果這一輪有收到座標，就進行判斷
        if current_pose:
            x, y, z = current_pose
            rospy.loginfo(f"  ...視覺系統偵測到物件在 [{x:.3f}, {y:.3f}, {z:.3f}]")

            # 判斷偵測到的座標是否在 handover 範圍內
            if (GRASP_X_MIN <= x <= GRASP_X_MAX and
                GRASP_Y_MIN <= y <= GRASP_Y_MAX and
                GRASP_Z_MIN <= z <= GRASP_Z_MAX):
                
                rospy.loginfo("✅ 物件已進入夾取範圍，鎖定座標並啟動抓取序列！")
                
                # 鎖定系統，不再接收新的視覺輸入
                with detection_lock:
                    is_detecting = False

                # 將這個觸發條件的座標，傳遞給我們之前寫好的、清晰的執行函數
                # (請確保您已經將上一主題中提供的 execute_grasp_and_handover_sequence 函數加入您的程式碼)
                success = execute_grasp_and_handover_sequence(current_pose, right_arm, left_arm, object_name)
                
                if success:
                    rospy.loginfo("--- 🎉 整個 handover 流程成功完成！---")
                else:
                    rospy.logerr("--- 😢 Handover 流程失敗或被中斷。---")
                
                # 無論成功或失敗，跳出這個監聽循環，等待下一次 main 函數的呼叫
                break
        
        rate.sleep()

def execute_grasp_and_handover_sequence(initial_target_pose, grasping_arm, receiving_arm, object_name):
    """
    執行一個完整的、基於單次偵測座標的抓取與交接序列。

    Args:
        initial_target_pose (tuple): (x, y, z) 格式的、經視覺偵測鎖定的物件世界座標。
        grasping_arm (MoveGroupCommander): 負責抓取物件的手臂 (例如全域的 right_arm)。
        receiving_arm (MoveGroupCommander): 負責接收物件的手臂 (例如全域的 left_arm)。
        object_name (str): 要操作的物件名稱，用於 link attacher。
    """
    global right_gripper, left_gripper, attach_srv, detach_srv

    rospy.loginfo(f"--- 🚀 開始為 '{object_name}' 執行完整的抓取與交接序列 ---")
    rospy.loginfo(f"🎯 初始目標鎖定於: [{initial_target_pose[0]:.3f}, {initial_target_pose[1]:.3f}, {initial_target_pose[2]:.3f}]")

    # --- 1. 定義所有動作的關鍵參數 (在這裡統一修改，方便調試) ---
    PRE_GRASP_HEIGHT = 0.13  # 物件上方預抓取點的高度偏移 (公尺)
    LIFT_HEIGHT = 0.13       # 抓取後抬升的高度 (公尺)
    
    # 交接點座標 (這裡是一個示例，請根據您的雙臂工作空間調整)
    # 這是右手拿著物件，移動到左手可以輕鬆接到的地方
    HANDOVER_POSE_RIGHT_ARM = (
        initial_target_pose[0] - 0.28, 
        initial_target_pose[1] - 0.10, 
        initial_target_pose[2] + 0.1 + LIFT_HEIGHT
    )
    # 這是左手準備去接物件的位置
    HANDOVER_POSE_LEFT_ARM = (
        initial_target_pose[0] - 0.28 + 0.023, # X軸稍微錯開
        initial_target_pose[1] - 0.10,       # Y軸對齊
        initial_target_pose[2] + 0.1 + LIFT_HEIGHT # Z軸對齊
    )

    # --- 2. 根據初始座標，計算出所有需要的目標點 ---
    grasp_pose = initial_target_pose
    grasp_pose = (grasp_pose[0], grasp_pose[1], grasp_pose[2] + 0.1)
    pre_grasp_pose = (grasp_pose[0], grasp_pose[1], grasp_pose[2] + PRE_GRASP_HEIGHT)
    post_grasp_lift_pose = (grasp_pose[0], grasp_pose[1], grasp_pose[2] + LIFT_HEIGHT)

    # --- 3. 執行序列化動作 ---
    try:
        # == 抓取階段 (由 grasping_arm 執行) ==
        rospy.loginfo(f"[1/6] 移動到預抓取位置: [{pre_grasp_pose[0]:.3f}, {pre_grasp_pose[1]:.3f}, {pre_grasp_pose[2]:.3f}]")
        move_arm_to_position(grasping_arm, pre_grasp_pose)
        
        rospy.loginfo(f"[2/6] 下降到抓取位置: [{grasp_pose[0]:.3f}, {grasp_pose[1]:.3f}, {grasp_pose[2]:.3f}]")
        move_arm_to_position(grasping_arm, grasp_pose)

        rospy.loginfo("[3/6] 執行夾取 (閉合夾爪並附加模型)...")
        # control_gripper(right_gripper, 0.01) # 根據您的夾爪模型調整閉合值
        # 建立 attach 請求
        req_grasp = AttachRequest()
        req_grasp.model_name_1 = object_name
        req_grasp.link_name_1 = "link"
        req_grasp.model_name_2 = "robot"
        req_grasp.link_name_2 = "rightarm_wrist_3_link" # 假設是右手抓取
        attach_srv.call(req_grasp)
        rospy.sleep(1.0) # 等待附加生效

        rospy.loginfo(f"[4/6] 抬升物件至: [{post_grasp_lift_pose[0]:.3f}, {post_grasp_lift_pose[1]:.3f}, {post_grasp_lift_pose[2]:.3f}]")
        move_arm_to_position(grasping_arm, post_grasp_lift_pose)
        
        # == 交接階段 ==
        rospy.loginfo(f"[5/6] 抓取臂移動到交接位置: [{HANDOVER_POSE_RIGHT_ARM[0]:.3f}, {HANDOVER_POSE_RIGHT_ARM[1]:.3f}, {HANDOVER_POSE_RIGHT_ARM[2]:.3f}]")
        move_arm_to_position(grasping_arm, HANDOVER_POSE_RIGHT_ARM)

        rospy.loginfo(f"[6/6] 接收臂移動到交接位置: [{HANDOVER_POSE_LEFT_ARM[0]:.3f}, {HANDOVER_POSE_LEFT_ARM[1]:.3f}, {HANDOVER_POSE_LEFT_ARM[2]:.3f}]")
        move_arm_to_position(receiving_arm, HANDOVER_POSE_LEFT_ARM)

        rospy.loginfo("--- ✅ 序列成功完成！準備執行物理交接 (detach/attach) ---")
        
        # 在這裡可以加入實際的 detach/attach 邏輯
        # detach_srv.call(req_grasp)
        # ...
        
        return True

    except Exception as e:
        rospy.logerr(f"❌ 在執行序列時發生錯誤: {e}")
        return False
    finally:
        # 無論成功或失敗，最後都重置回偵測階段，準備下一次任務
        rospy.loginfo("--- 序列結束，重置回偵測模式 ---")
        reset_to_detection()


def main():

    global arm_mover

    # 初始化 MoveIt, ur3手臂和夾爪
    initialize_moveit()
    initialize_gripper()
    initialize_link_attacher()

    # arm_mover = ArmMover()
    # arm_mover.start_threads()  # 開始運行手臂移動執行緒
    
    # move_hammer(right_arm, right_arm_gripper)

    # attach_hammer(left_arm, left_arm_gripper, right_arm, right_arm_gripper, attach_srv, detach_srv)

    # 設定物件資訊
    object_name = "hammer"  # 你要在 Gazebo 內的名稱
    object_path = os.path.expanduser("~/ros_ws/src/ros_ur3/ur_gripper_gazebo/models/hammer/model.sdf")  # 替換成你的物件 URDF/SDF 路徑
    x, y, z = 1.0, 0.15, 0.7  # 物件位置
    roll, pitch, yaw = 0, 0, 0  # 旋轉角度

    # attach_hammer(left_arm, left_gripper, right_arm, right_gripper, attach_srv, detach_srv)
    spawn_sdf_object(object_name, object_path, x, y, z, roll, pitch, yaw)
    # is_object_in_handover_area(object_name)
    vision_based_handover_trigger(object_name)

    rospy.sleep(2)  # 確保 pose 發送啟動

    # 清理並結束
    moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
