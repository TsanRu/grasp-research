#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import subprocess
import rospy
import threading
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import RobotTrajectory, RobotState
from gazebo_ros_link_attacher.srv import Attach, AttachRequest


# ==============================================================================
# --- 設定 AI 橋接參數 (請修改這裡！) ---
# ==============================================================================
# 1. 您的虛擬環境 Python 路徑 (請用 `which python` 在 gemini_env 下確認)
AI_PYTHON_PATH = "/home/rvl/miniconda3/envs/gemini_env/bin/python" 

# 2. 您的 ai_planner.py 絕對路徑
AI_SCRIPT_PATH = "/home/rvl/ros_ws/src/ros_ur3/ur_control/scripts/ai_planner.py"


# ==============================================================================
# --- 步驟一: AI 任務翻譯官 (Adapter) ---
# ==============================================================================
def get_ai_tasks_and_translate(image_path, instruction, pose_lookup_table):
    """
    呼叫 AI 規劃器，過濾雜訊，並將回傳的簡化 JSON 轉換為可執行格式。
    """
    rospy.loginfo(f"正在呼叫 AI 大腦... 指令: {instruction}")
    
    # 0. 基礎檢查
    if not os.path.exists(AI_PYTHON_PATH):
        rospy.logerr(f"找不到 AI Python 環境: {AI_PYTHON_PATH}")
        return []
    if not os.path.exists(AI_SCRIPT_PATH):
        rospy.logerr(f"找不到 AI 腳本: {AI_SCRIPT_PATH}")
        return []

    try:
        # 1. 透過 subprocess 呼叫獨立環境的 AI script (使用 Popen 以分離 stdout/stderr)
        process = subprocess.Popen(
            [AI_PYTHON_PATH, AI_SCRIPT_PATH, image_path, instruction],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        # 2. 【關鍵修改】雜訊過濾邏輯
        raw_output = stdout.decode('utf-8').strip()
        
        # 尋找 JSON 的開始 '[' 和結束 ']'
        start_idx = raw_output.find('[')
        end_idx = raw_output.rfind(']') + 1

        if start_idx == -1 or end_idx == 0:
            rospy.logerr(f"AI 回傳資料中找不到 JSON 列表。原始輸出:\n{raw_output}")
            rospy.logerr(f"錯誤訊息 (stderr): {stderr.decode('utf-8')}")
            return []

        # 只截取 JSON 的部分
        clean_json_str = raw_output[start_idx:end_idx]
        ai_json = json.loads(clean_json_str)

    except Exception as e:
        rospy.logerr(f"AI 執行失敗或解析錯誤: {e}")
        return []

    rospy.loginfo("AI 規劃成功，開始翻譯任務參數...")
    
    real_tasks = []
    
    # 通用參數
    common_params = {'model': "hammer", 'link': "link", 'robot': "robot"}

    # 輔助函式：快速生成 params
    def make_params(arm):
        p = common_params.copy()
        p['gripper_link'] = 'leftarm_wrist_3_link' if arm == 'left' else 'rightarm_wrist_3_link'
        return p

    # 3. 遍歷 AI 指令並翻譯
    for step in ai_json:
        if 'thought' in step:
            rospy.loginfo(f"AI 思路: {step['thought']}")
            continue

        new_task = {'desc': step.get('desc', 'AI Task'), 'type': step['type']}

        # 翻譯 Pose
        if step['type'] == 'dual_p2p':
            if 'left' in step: new_task['left_pose'] = pose_lookup_table[step['left']]
            if 'right' in step: new_task['right_pose'] = pose_lookup_table[step['right']]

        elif 'cartesian' in step['type'] or 'p2p' in step['type']:
            if 'waypoints' in step:
                wps = []
                for wp in step['waypoints']:
                    if wp in pose_lookup_table:
                        wps.append(pose_lookup_table[wp])
                    else:
                        rospy.logwarn(f"未知座標點: {wp}")
                new_task['waypoints'] = wps

        elif step['type'] == 'action':
            new_task['action'] = step.get('action', step.get('act'))
            new_task['params'] = make_params(step.get('arm'))

        real_tasks.append(new_task)

    return real_tasks


# -------------------------------------------------------------------
# --- 步驟二: 升級我們的中央調度器類別，使其支持軌跡拼接 ---
# -------------------------------------------------------------------
class TaskDispatcher:
    def __init__(self, left_group, right_group, both_arms_group, robot_commander):
        self.left_group = left_group
        self.right_group = right_group
        self.both_arms_group = both_arms_group
        self.robot = robot_commander
        self.left_eef_link = self.left_group.get_end_effector_link()
        self.right_eef_link = self.right_group.get_end_effector_link()
        try:
            self.attach_srv = rospy.ServiceProxy('/link_attacher_node/attach', Attach)
            self.detach_srv = rospy.ServiceProxy('/link_attacher_node/detach', Attach)
            self.attach_srv.wait_for_service(timeout=5.0)
            self.detach_srv.wait_for_service(timeout=5.0)
        except Exception as e:
            rospy.logerr(f"連接 Attach/Detach 服務失敗: {e}"); raise
        self.execution_queue = []
        self.current_robot_state = self.robot.get_current_state()
        rospy.loginfo("中央任務調度器 (TaskDispatcher) 初始化完畢。")
        
    def set_speed_scaling(self, speed_factor=0.1, accel_factor=0.1):
        # ... 此函式不變 ...
        self.left_group.set_max_velocity_scaling_factor(speed_factor)
        self.right_group.set_max_velocity_scaling_factor(speed_factor)
        self.both_arms_group.set_max_velocity_scaling_factor(speed_factor)
        self.left_group.set_max_acceleration_scaling_factor(accel_factor)
        self.right_group.set_max_acceleration_scaling_factor(accel_factor)
        self.both_arms_group.set_max_acceleration_scaling_factor(accel_factor)
        rospy.loginfo(f"所有規劃群組速度比例已設為: {speed_factor}")
        
    def _create_robust_trajectory(self, trajectories):
        """
        將多個【未計時的】分段笛卡兒路徑拼接成一個單一、手動計時的軌跡。
        """
        if not trajectories: return None
        joint_names = trajectories[0].joint_trajectory.joint_names
        for traj in trajectories[1:]:
            if traj.joint_trajectory.joint_names != joint_names:
                rospy.logerr("軌跡拼接失敗：關節名稱不一致！")
                return None
        final_trajectory = RobotTrajectory()
        final_trajectory.joint_trajectory.header = trajectories[0].joint_trajectory.header
        final_trajectory.joint_trajectory.joint_names = joint_names
        all_points = []
        if trajectories:
            all_points.extend(trajectories[0].joint_trajectory.points)
            for traj in trajectories[1:]:
                all_points.extend(traj.joint_trajectory.points[1:])
        current_time, time_step = 0.0, 0.1
        for point in all_points:
            point.time_from_start = rospy.Duration.from_sec(current_time)
            point.velocities, point.accelerations = [], []
            final_trajectory.joint_trajectory.points.append(point)
            current_time += time_step
        return final_trajectory

    def plan_task_script(self, tasks):
        # --- 為了更穩健，我們在這裡處理 MoveIt API 的返回不一致問題 ---
        rospy.loginfo("="*20 + " 中央調度器開始預規劃所有任務 " + "="*20)
        for task in tasks:
            rospy.loginfo(f"--- 正在規劃任務: {task['desc']} ---")
            task_type, plan, group = task['type'], None, None
            
            # 清理殘留目標
            self.left_group.clear_pose_targets(); self.right_group.clear_pose_targets(); self.both_arms_group.clear_pose_targets()
            
            if task_type == 'dual_p2p':
                group = self.both_arms_group
                group.set_start_state(self.current_robot_state)
                group.set_pose_target(task['left_pose'], self.left_eef_link)
                group.set_pose_target(task['right_pose'], self.right_eef_link)
                plan = group.plan()
                if isinstance(plan, tuple): plan = plan[1] # 處理 plan() 返回 tuple 的情況
            
            elif task_type in ['right_p2p', 'left_p2p']:
                group = self.right_group if 'right' in task_type else self.left_group
                group.set_start_state(self.current_robot_state)
                group.set_pose_target(task['waypoints'][0])
                plan = group.plan()
                if isinstance(plan, tuple): plan = plan[1]

            elif task_type in ['right_cartesian', 'left_cartesian']:
                group = self.right_group if 'right' in task_type else self.left_group
                group.set_start_state(self.current_robot_state)
                # 注意：compute_cartesian_path 第三個參數 avoid_collisions 設為 True
                (plan, fraction) = group.compute_cartesian_path(task['waypoints'], 0.01, True)
                if fraction < 0.95: rospy.logerr(f"笛卡兒規劃失敗(完成度:{fraction*100:.1f}%)"); return False
            
            elif task_type == 'action':
                self.execution_queue.append({'type': 'action', 'task': task})
                rospy.loginfo("離散動作已加入佇列。")
                continue

            if plan and plan.joint_trajectory.points:
                self.execution_queue.append({'type': 'plan', 'plan': plan, 'group': group, 'desc': task['desc']})
                rospy.loginfo("軌跡規劃成功，已加入佇列。")
                self._update_robot_state(plan)
            else:
                rospy.logerr(f"任務 '{task['desc']}' 未能生成有效軌跡。"); return False
        
        rospy.loginfo("="*20 + " 所有任務預規劃成功！ " + "="*20)
        return True

    def execute(self):
        """
        【終極版】執行佇列，並智能地將由相同 group 規劃的連續運動拼接起來。
        """
        if not self.execution_queue:
            rospy.logwarn("執行序列為空。"); return

        rospy.loginfo("="*20 + " 開始執行智能拼接式任務序列 " + "="*20)
        
        motion_chunk = []
        for item in self.execution_queue:
            if rospy.is_shutdown(): break
            
            if item['type'] == 'plan':
                # 【核心邏輯】檢查當前 plan 是否可以拼接到現有的 motion_chunk 中
                if not motion_chunk or item['group'] == motion_chunk[0]['group']:
                    # 如果塊是空的，或者當前 plan 的規劃組與塊中第一個 plan 的組相同，則加入
                    motion_chunk.append(item)
                else:
                    # 規劃組不同！這是一個拼接“打斷點”。
                    # 1. 先執行並清空舊的運動塊
                    self._execute_motion_chunk(motion_chunk)
                    # 2. 用當前 plan 開始一個新的運動塊
                    motion_chunk = [item]
            
            elif item['type'] == 'action':
                # 離散動作永遠是“打斷點”。
                # 1. 先執行並清空之前可能存在的運動塊
                if motion_chunk:
                    self._execute_motion_chunk(motion_chunk)
                    motion_chunk = []
                
                # 2. 再執行離散動作
                self._execute_action(item['task'])
        
        # 迴圈結束後，別忘了處理最後一個未執行的運動塊
        if motion_chunk and not rospy.is_shutdown():
            self._execute_motion_chunk(motion_chunk)

        rospy.loginfo("="*20 + " 所有任務執行完畢！ " + "="*20)

    def _execute_motion_chunk(self, chunk):
        if not chunk: return
        
        execution_group = chunk[0]['group']
        rospy.loginfo(f"--- 準備執行一個包含 {len(chunk)} 段運動的運動塊，執行者: [{execution_group.get_name()}] ---")

        
        # 收集 chunk 裡的所有任務描述並印出
        task_names = " -> ".join([item['desc'] for item in chunk])
        rospy.loginfo(f"--- 執行任務 (軌跡) : [{task_names}] ---")
        
        trajectories = [item['plan'] for item in chunk]
        first_traj = trajectories[0]

        # 智能檢測：如果軌跡點數大於1，且第二個點的時間戳大於0，則認為是“已計時”軌跡
        is_timed = len(first_traj.joint_trajectory.points) > 1 and first_traj.joint_trajectory.points[1].time_from_start.to_sec() > 0.0

        if is_timed:
            # 對於來自 plan() 的已計時軌跡，分段執行以尊重 MoveIt 的速度規劃
            rospy.loginfo("檢測到已計時軌跡 (來自 plan())。將分段執行以確保速度控制。")
            for i, plan in enumerate(trajectories):
                rospy.loginfo(f"  > 執行分段 {i+1}/{len(trajectories)}...")
                execution_group.execute(plan, wait=True)
            execution_group.stop()
        else:
            # 對於來自 compute_cartesian_path() 的未計時軌跡，使用您的拼接函式賦予時間
            rospy.loginfo("檢測到未計時軌跡 (來自 compute_cartesian_path())。將進行拼接和手動計時。")
            stitched_plan = self._create_robust_trajectory(trajectories)
            if stitched_plan:
                execution_group.execute(stitched_plan, wait=True)
                execution_group.stop()
            else:
                rospy.logerr("軌跡拼接失敗，此運動塊執行已取消！")

    def _update_robot_state(self, plan):
        last_point = plan.joint_trajectory.points[-1]
        joint_map = dict(zip(plan.joint_trajectory.joint_names, last_point.positions))
        current_joint_names = self.current_robot_state.joint_state.name
        new_positions = list(self.current_robot_state.joint_state.position)
        for i, name in enumerate(current_joint_names):
            if name in joint_map: new_positions[i] = joint_map[name]
        self.current_robot_state.joint_state.position = tuple(new_positions)
        rospy.loginfo("--- 內部機器人狀態已更新 ---")

    def _execute_action(self, task):
        action, params = task['action'], task['params']
        rospy.loginfo(f"--- 執行任務 (動作) : {task['desc']} ({action.upper()}) ---")
        req = AttachRequest(model_name_1=params['model'], link_name_1=params['link'], model_name_2=params['robot'], link_name_2=params['gripper_link'])
        if action == 'grasp': self.attach_srv.call(req)
        elif action == 'release': self.detach_srv.call(req)
        rospy.sleep(1.0)


# ---------------------------------------------------------------
# --- 步驟三: 場景輔助函式 ---
# ---------------------------------------------------------------
def create_pose(x, y, z, ox=0.0, oy=0.0, oz=0.0, ow=1.0):
    p = Pose(); p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = ox, oy, oz, ow
    return p

def add_ur_base_to_scene(scene, name_prefix, base_pose_x, base_pose_y, base_pose_z=0):
    rospy.loginfo(f"Adding {name_prefix} to the planning scene...")
    # 頂板
    plate_pose = PoseStamped()
    plate_pose.header.frame_id = "world"
    plate_pose.pose.position.x = base_pose_x + 0.0
    plate_pose.pose.position.y = base_pose_y + 0.0
    plate_pose.pose.position.z = base_pose_z + 0.67501
    plate_pose.pose.orientation.w = 1.0
    scene.add_box(f"{name_prefix}_top_plate", plate_pose, size=(0.5, 0.5, 0.02))
    leg_size = (0.04, 0.04, 0.655)
    leg_relative_z = 0.35
    leg_positions = {
        "leg1": (0.22, 0.22),
        "leg2": (-0.22, 0.22),
        "leg3": (0.22, -0.22),
        "leg4": (-0.22, -0.22),
    }
    for leg_name, (rel_x, rel_y) in leg_positions.items():
        leg_pose = PoseStamped()
        leg_pose.header.frame_id = "world"
        leg_pose.pose.position.x = base_pose_x + rel_x
        leg_pose.pose.position.y = base_pose_y + rel_y
        leg_pose.pose.position.z = base_pose_z + leg_relative_z
        leg_pose.pose.orientation.w = 1.0
        scene.add_box(f"{name_prefix}_{leg_name}", leg_pose, size=leg_size)

def add_default_scene_objects(scene):
    add_ur_base_to_scene(scene, "ur_base_left",   0.775, -0.5)
    add_ur_base_to_scene(scene, "ur_base_right",  0.775,  0.5)
    add_ur_base_to_scene(scene, "ur_base_middle", 0.775,  0.0)
    add_ur_base_to_scene(scene, "ur_base_top_left",    0.275, -0.5)
    add_ur_base_to_scene(scene, "ur_base_top_right",   0.275,  0.5)
    add_ur_base_to_scene(scene, "ur_base_top_middle",  0.275,  0.0)
    add_ur_base_to_scene(scene, "ur_base_down_left",   1.275, -0.5)
    add_ur_base_to_scene(scene, "ur_base_down_right",  1.275,  0.5)
    add_ur_base_to_scene(scene, "ur_base_down_middle", 1.275,  0.0)
    rospy.loginfo("已將所有基座模型加入Planning Scene作為避障")
    rospy.sleep(1)


# ==============================================================================
# --- 步驟四: Main Function (接上大腦！) ---
# ==============================================================================

if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('central_dispatcher_node', anonymous=True)

    # 1. 初始化 MoveIt 元件
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    left_group = moveit_commander.MoveGroupCommander("leftarm")
    right_group = moveit_commander.MoveGroupCommander("rightarm")
    both_arms_group = moveit_commander.MoveGroupCommander("both_arms")

    rospy.sleep(2)    # Scene初始化需留2秒（很重要）。
    add_default_scene_objects(scene)
    
    # 2. 建立我們的中央調度器實例
    dispatcher = TaskDispatcher(left_group, right_group, both_arms_group, robot)
    # dispatcher.set_speed_scaling(speed_factor=0.3, accel_factor=0.3)

    # 3. 編寫您的「劇本」
    rospy.loginfo("="*20 + " 定義雙臂任務序列 " + "="*20)
    # 定義一些關鍵姿態
    grasp_approach_pose = create_pose(1.0, 0.15, 1.0, oy=1.0, ow=0.0)
    grasp_pose = create_pose(1.0, 0.15, 0.85, oy=1.0, ow=0.0)
    handover_pose_r = create_pose(0.76, 0.15, 1.0, oy=1.0, ow=0.0)
    handover_approach_pose_l = create_pose(0.76, -0.05, 1.1, oy=1.0, ow=0.0)
    handover_pose_l = create_pose(0.76, -0.05, 0.97, oy=1.0, ow=0.0)
    retreat_pose_r = create_pose(1.0, 0.15, 1.1, oy=1.0, ow=0.0)
    retreat_pose_l = create_pose(0.65, -0.05, 1.1, oy=1.0, ow=0.0)

    # OBJECT_MODEL, OBJECT_LINK, ROBOT_MODEL = "hammer", "link", "robot"
    
    # tasks = [
    #     {'desc': '雙臂移動到抓取預備區', 'type': 'dual_p2p', 'left_pose': handover_approach_pose_l, 'right_pose': grasp_approach_pose},
    #     # 【注意】這裡的兩個 cartesian 任務是連續的，它們會被拼接在一起執行！
    #     {'desc': '右臂下降抓取(直線)', 'type': 'right_cartesian', 'waypoints': [grasp_pose]},
    #     {'desc': '執行抓取', 'type': 'action', 'action': 'grasp', 'params': {'model': OBJECT_MODEL, 'link': OBJECT_LINK, 'robot': ROBOT_MODEL, 'gripper_link': 'rightarm_wrist_3_link'}},
    #     {'desc': '右臂抬起物體(直線)', 'type': 'right_cartesian', 'waypoints': [grasp_approach_pose]},
    #     {'desc': '右臂移動到交接點', 'type': 'right_p2p', 'waypoints': [handover_pose_r]},
    #     {'desc': '左臂移動到交接夾取點', 'type': 'left_cartesian', 'waypoints': [handover_pose_l]},
    #     {'desc': '左臂抓取物體', 'type': 'action', 'action': 'grasp', 'params': {'model': OBJECT_MODEL, 'link': OBJECT_LINK, 'robot': ROBOT_MODEL, 'gripper_link': 'leftarm_wrist_3_link'}},
    #     {'desc': '右臂釋放物體', 'type': 'action', 'action': 'release', 'params': {'model': OBJECT_MODEL, 'link': OBJECT_LINK, 'robot': ROBOT_MODEL, 'gripper_link': 'rightarm_wrist_3_link'}},
    #     # {'desc': '雙臂同時安全撤離', 'type': 'dual_p2p', 'left_pose': retreat_pose_l, 'right_pose': retreat_pose_r},
    #     {'desc': '右臂安全撤離', 'type': 'right_p2p', 'waypoints': [retreat_pose_r]},
    #     {'desc': '左臂安全撤離', 'type': 'left_p2p', 'waypoints': [retreat_pose_l]},
    #     {'desc': '左臂釋放物體', 'type': 'action', 'action': 'release', 'params': {'model': OBJECT_MODEL, 'link': OBJECT_LINK, 'robot': ROBOT_MODEL, 'gripper_link': 'leftarm_wrist_3_link'}},
    # ]

    # # 4. 指揮調度器規劃並執行
    # if dispatcher.plan_task_script(tasks):
    #     dispatcher.execute()
        
        
    POSE_LOOKUP = {
        "grasp_approach_pose": grasp_approach_pose,
        "grasp_pose": grasp_pose,
        "handover_approach_pose_l": handover_approach_pose_l,
        "handover_pose_r": handover_pose_r,
        "handover_pose_l": handover_pose_l,
        "retreat_pose_l": retreat_pose_l,
        "retreat_pose_r": retreat_pose_r,
        # 記得補上 AI 可能會用到的
        "handover_approach_pose_r": create_pose(0.76, 0.08, 1.1, oy=1.0, ow=0.0) 
    }

    # 4. 準備輸入
    current_image = "test_scene.png"  # 確保這張圖存在
    user_instruction = "將物件從右手交給左手" # 這是給 AI 的指令

    # ==========================================
    # ★ 關鍵修改：呼叫 AI 翻譯官，而不是用寫死的 tasks
    # ==========================================
    
    # 呼叫 Step 1 的函式，取得 AI 規劃好的任務
    final_tasks = get_ai_tasks_and_translate(current_image, user_instruction, POSE_LOOKUP)

    # 5. 執行
    if final_tasks:
        rospy.loginfo(f"AI 任務生成成功，共 {len(final_tasks)} 步，開始執行...")
        if dispatcher.plan_task_script(final_tasks):
            dispatcher.execute()
    else:
        rospy.logwarn("AI 任務生成失敗，請檢查 API Key 或 python 路徑")

    moveit_commander.roscpp_shutdown()
