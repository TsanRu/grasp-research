#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import json
import threading
import open3d as o3d

ros_path = '/opt/ros/noetic/lib/python3/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append(ros_path)

try:
    import rospy
    import message_filters
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    from geometry_msgs.msg import Pose
    import tf2_ros
    print("✅ 成功跨界連接 ROS Noetic！")
except ImportError:
    print("❌ 找不到 ROS")

import numpy as np
from scipy.spatial.transform import Rotation
from gsnet import AnyGrasp

# ─────────────────────────────────────────────────────────────────────────────
# 術語定義：
#   接收臂 (receiver arm) = 左臂，末端裝有 Hand-in-Eye 相機
#   操作臂 (operator arm) = 右臂，負責從桌面夾取並搬運物件
# ─────────────────────────────────────────────────────────────────────────────


def imgmsg_to_numpy(msg):
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


class Config:
    def __init__(self):
        self.checkpoint_path = './log/checkpoint_detection.tar'
        self.max_gripper_width = 0.085
        self.gripper_height = 0.04
        self.top_down_grasp = False
        self.debug = False
        # 從環境變數讀取，預設開啟 debug；正式執行時 export ANYGRASP_DEBUG=0 關閉
        self.debug = os.environ.get("ANYGRASP_DEBUG", "1") == "1"


class AnyGraspHandoverNode:
    def __init__(self):
        rospy.init_node('anygrasp_handover_node', anonymous=True)
        self.cfgs = Config()
        self.fx, self.fy = 462.16757, 462.16757
        self.cx, self.cy = 320.5, 240.5
        self.mask_dir = "/home/rvl/ros_ws/src/anygrasp_sdk/grasp_detection/my_gazebo_data"

        rospy.loginfo("🦾 正在載入 AnyGrasp...")
        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        rospy.loginfo("✅ AnyGrasp 載入完成！")

        self.color_sub = message_filters.Subscriber(
            '/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber(
            '/camera/aligned_depth_to_color/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber("/system/trigger_detection", String,
                         self.trigger_callback)

        self.plan_pub = rospy.Publisher(
            "/anygrasp/handover_plan", String, queue_size=1)

        self.need_detection = False
        self.target_object = "unknown_object"
        self.mode = "operator_only"  # "dual" 或 "receiver_only"

        rospy.loginfo("🤖 AnyGrasp 節點就緒，等待觸發...")

    def trigger_callback(self, msg):
        # 支援兩種格式：純字串（舊格式）或 JSON（新格式含 mode）
        try:
            data = json.loads(msg.data)
            self.target_object = data.get("object_name", "unknown")
            self.mode = data.get("mode", "operator_only")
        except (json.JSONDecodeError, TypeError):
            self.target_object = msg.data
            self.mode = "operator_only"

        rospy.loginfo(
            f"⚡ 收到 Trigger！物件: {self.target_object}，模式: {self.mode}")
        self.need_detection = True

    def to_ros_pose(self, grasp):
        p = Pose()
        p.position.x, p.position.y, p.position.z = grasp.translation
        q = Rotation.from_matrix(grasp.rotation_matrix).as_quat()
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    def pose_to_dict(self, pose):
        return {
            'position': {
                'x': pose.position.x,
                'y': pose.position.y,
                'z': pose.position.z
            },
            'orientation': {
                'x': pose.orientation.x,
                'y': pose.orientation.y,
                'z': pose.orientation.z,
                'w': pose.orientation.w
            }
        }

    def get_arm_base_position(self, arm_name, ref_frame):
        # arm_name 使用 ROS TF 名稱：'rightarm' 或 'leftarm'
        try:
            base_link_name = f"{arm_name}_base_link"
            trans = self.tf_buffer.lookup_transform(
                ref_frame, base_link_name, rospy.Time(0), rospy.Duration(1.0))
            return np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
        except Exception:
            if "left" in arm_name:
                return np.array([-0.5, 0.2, 0.0])
            else:
                return np.array([0.5, 0.2, 0.0])

    def filter_outlier_points(self, pts):
        if pts.shape[0] < 20:
            return pts
        q1 = np.percentile(pts[:, 2], 25)
        q3 = np.percentile(pts[:, 2], 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        mask = (pts[:, 2] >= lower_bound) & (pts[:, 2] <= upper_bound)
        clean_pts = pts[mask]
        if clean_pts.shape[0] < 20 or iqr < 0.01:
            z_median = np.median(pts[:, 2])
            mask_fallback = np.abs(pts[:, 2] - z_median) < 0.10
            return pts[mask_fallback]
        return clean_pts

    def get_dynamic_lims(self, points_target, pad=0.0, y_top_pad=0.05):
        if points_target.shape[0] < 20:
            return None
        min_b = points_target.min(axis=0)
        max_b = points_target.max(axis=0)
        return [
            min_b[0] - pad, max_b[0] + pad,
            min_b[1] - y_top_pad, max_b[1] + pad,
            min_b[2] - pad, max_b[2] + pad
        ]

    def get_arm_specific_grasps(self, gg, arm_name, ref_frame):
        # arm_name 使用 ROS TF 名稱：'rightarm'（操作臂）或 'leftarm'（接收臂）
        arm_base_pos = self.get_arm_base_position(arm_name, ref_frame)
        arm_grasps = []
        for i in range(len(gg)):
            grasp = gg[i]
            grasp_approach = grasp.rotation_matrix[:, 0]

            if "right" in arm_name:  # 操作臂
                if grasp_approach[0] > 0.5:
                    continue
                if grasp_approach[1] > 0.8:
                    continue
            else:  # 接收臂
                if grasp_approach[0] < -0.3:
                    continue
                if grasp_approach[1] > 0.8:
                    continue

            dist = np.linalg.norm(grasp.translation - arm_base_pos)

            if "left" in arm_name:  # 接收臂
                side_bonus = grasp_approach[0]
            else:  # 操作臂
                side_bonus = -grasp_approach[0]

            top_bonus = -grasp_approach[1] * 0.5
            arm_score = (5.0 + grasp.score
                         + side_bonus * 2.0
                         + top_bonus
                         - dist * 8.0)

            arm_grasps.append({
                'orig_idx': i,
                'grasp': grasp,
                'arm_score': arm_score
            })

        arm_grasps.sort(key=lambda x: x['arm_score'], reverse=True)
        return arm_grasps

    def generate_handover_plan(self, operator_grasps, receiver_grasps):
        if not operator_grasps or not receiver_grasps:
            return []

        valid_pairs = []
        for op_item in operator_grasps:
            for recv_item in receiver_grasps:
                grasp_op = op_item['grasp']
                grasp_recv = recv_item['grasp']
                pos_op = grasp_op.translation
                pos_recv = grasp_recv.translation

                dist = np.linalg.norm(pos_op - pos_recv)

                vec_op = grasp_op.rotation_matrix[:, 0]
                vec_recv = grasp_recv.rotation_matrix[:, 0]
                cos_theta = np.clip(np.dot(vec_op, vec_recv), -1.0, 1.0)
                angle_rad = np.arccos(cos_theta)

                # 夾角小於 60 度表示接收臂與操作臂接近方向太相似，容易互撞
                # if angle_rad < np.radians(60):
                #     continue

                valid_pairs.append({
                    'op_item': op_item,
                    'recv_item': recv_item,
                    'dist': dist,
                    'angle_rad': angle_rad,
                    'pos_op': pos_op
                })

        if not valid_pairs:
            return []

        dists = np.array([p['dist'] for p in valid_pairs])
        mu_dist = np.mean(dists)
        std_dist = np.std(dists) + 1e-6

        w_dist = 10.0
        w_angle = 3.0
        w_height = 10.0
        w_arm_self = 2.0

        grouped_pairs = {}
        for p in valid_pairs:
            z_dist = (p['dist'] - mu_dist) / std_dist
            score_height = p['pos_op'][2] * w_height
            total_score = (p['op_item']['arm_score']
                           + p['recv_item']['arm_score']
                           + w_dist * z_dist
                           + w_angle * p['angle_rad']
                           + score_height)

            op_idx = p['op_item']['orig_idx']
            if op_idx not in grouped_pairs:
                grouped_pairs[op_idx] = {
                    'orig_idx_R': op_idx,
                    'pose_R': self.to_ros_pose(p['op_item']['grasp']),
                    'r_arm_score': p['op_item']['arm_score'],
                    'max_pair_score': -float('inf'),
                    'left_candidates': []
                }

            grouped_pairs[op_idx]['left_candidates'].append({
                'orig_idx_L': p['recv_item']['orig_idx'],
                'score': total_score,
                'pose_L': self.to_ros_pose(p['recv_item']['grasp'])
            })

            if total_score > grouped_pairs[op_idx]['max_pair_score']:
                grouped_pairs[op_idx]['max_pair_score'] = total_score

        for op_idx in grouped_pairs:
            grouped_pairs[op_idx]['left_candidates'].sort(
                key=lambda x: x['score'], reverse=True)

        ranked_groups = list(grouped_pairs.values())
        for g in ranked_groups:
            g['group_score'] = (g['r_arm_score'] * w_arm_self
                                + g['max_pair_score'])
        ranked_groups.sort(key=lambda x: x['group_score'], reverse=True)

        json_plan = []
        for group in ranked_groups:
            plan_item = {
                'orig_idx_R': group['orig_idx_R'],
                'group_score': group['group_score'],
                'pose_R_table': self.pose_to_dict(group['pose_R']),
                'left_candidates': []
            }
            for cand in group['left_candidates']:
                plan_item['left_candidates'].append({
                    'orig_idx_L': cand['orig_idx_L'],
                    'score': cand['score'],
                    'pose_L_table': self.pose_to_dict(cand['pose_L'])
                })
            json_plan.append(plan_item)

        return json_plan

    def generate_operator_plan(self, operator_grasps, camera_frame):
        """生成操作臂（右臂）從桌面夾取的計畫書"""
        if not operator_grasps:
            return []

        json_plan = []
        for i, op_item in enumerate(operator_grasps):
            pose_op = self.to_ros_pose(op_item['grasp'])
            pose_op_dict = self.pose_to_dict(pose_op)
            pose_op_world = self.transform_grasp_to_world(pose_op_dict, camera_frame)
            plan_item = {
                'orig_idx_R': i,
                'group_score': op_item['arm_score'],
                'pose_R_table': pose_op_world,
                'left_candidates': []
            }
            json_plan.append(plan_item)
        return json_plan

    def generate_receiver_plan(self, receiver_grasps, camera_frame):
        """生成接收臂（左臂）接取的計畫書"""
        if not receiver_grasps:
            return []

        json_plan = []
        for i, recv_item in enumerate(receiver_grasps):
            pose_recv = self.to_ros_pose(recv_item['grasp'])
            pose_recv_dict = self.pose_to_dict(pose_recv)
            pose_recv_world = self.transform_grasp_to_world(pose_recv_dict, camera_frame)
            plan_item = {
                'orig_idx_R': i,
                'group_score': recv_item['arm_score'],
                'pose_R_table': pose_recv_world,
                'left_candidates': [{
                    'orig_idx_L': recv_item['orig_idx'],
                    'score': recv_item['arm_score'],
                    'pose_L_table': pose_recv_world
                }]
            }
            json_plan.append(plan_item)
        return json_plan

    def transform_grasp_to_world(self, grasp_pose_dict, camera_frame):
        """將相機座標系下的夾取姿態轉換到世界座標系"""
        try:
            trans = self.tf_buffer.lookup_transform(
                "world",
                camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            t = trans.transform.translation
            q = trans.transform.rotation
            T = np.eye(4)
            T[:3, 3] = [t.x, t.y, t.z]
            T[:3, :3] = Rotation.from_quat(
                [q.x, q.y, q.z, q.w]).as_matrix()

            pos_cam = np.array([
                grasp_pose_dict['position']['x'],
                grasp_pose_dict['position']['y'],
                grasp_pose_dict['position']['z'],
                1.0
            ])
            pos_world = T @ pos_cam

            rot_cam = Rotation.from_quat([
                grasp_pose_dict['orientation']['x'],
                grasp_pose_dict['orientation']['y'],
                grasp_pose_dict['orientation']['z'],
                grasp_pose_dict['orientation']['w']
            ]).as_matrix()
            rot_world = T[:3, :3] @ rot_cam
            q_world = Rotation.from_matrix(rot_world).as_quat()

            return {
                'position': {
                    'x': float(pos_world[0]),
                    'y': float(pos_world[1]),
                    'z': float(pos_world[2])
                },
                'orientation': {
                    'x': float(q_world[0]),
                    'y': float(q_world[1]),
                    'z': float(q_world[2]),
                    'w': float(q_world[3])
                }
            }
        except Exception as e:
            rospy.logwarn(f"⚠️ TF 轉換失敗，使用原始相機座標: {e}")
            return grasp_pose_dict

    def callback(self, color_msg, depth_msg):
        if not self.need_detection:
            return

        try:
            self.need_detection = False
            mode = self.mode
            rospy.loginfo(f"⚙️ 啟動偵測管線，模式: {mode}")

            color_np = imgmsg_to_numpy(color_msg)
            if "bgr8" in color_msg.encoding:
                color_np = color_np[:, :, ::-1]
            colors = color_np.astype(np.float32) / 255.0

            depths = imgmsg_to_numpy(depth_msg).astype(np.float32)
            max_depth_val = np.nanmax(depths) if np.nanmax(depths) > 0 else 0.0
            scale = 1000.0 if max_depth_val > 100 else 1.0

            receiver_mask_path = os.path.join(self.mask_dir, "receiver_mask.png")
            operator_mask_path = os.path.join(self.mask_dir, "operator_mask.png")

            if mode == "receiver_only":
                if not os.path.exists(receiver_mask_path):
                    rospy.logerr("❌ 找不到 receiver_mask")
                    return
            elif mode == "operator_only":
                if not os.path.exists(operator_mask_path):
                    rospy.logerr("❌ 找不到 operator_mask")
                    return
            else:
                rospy.logwarn(f"⚠️ 未知模式 '{mode}'，預設使用 operator_only")
                mode = "operator_only"
                if not os.path.exists(operator_mask_path):
                    rospy.logerr("❌ 找不到 operator_mask")
                    return

            xmap, ymap = np.meshgrid(
                np.arange(depths.shape[1]), np.arange(depths.shape[0]))
            points_z = depths / scale
            points_x = (xmap - self.cx) / self.fx * points_z
            points_y = (ymap - self.cy) / self.fy * points_z

            valid_depth_mask = (points_z > 0)
            points_full = np.stack(
                [points_x, points_y, points_z],
                axis=-1)[valid_depth_mask].astype(np.float32)
            colors_full = colors[valid_depth_mask].astype(np.float32)

            MAX_POINTS = 30000
            if points_full.shape[0] > MAX_POINTS:
                indices = np.random.choice(
                    points_full.shape[0], MAX_POINTS, replace=False)
                points_full = points_full[indices]
                colors_full = colors_full[indices]

            camera_frame = (color_msg.header.frame_id
                            if color_msg.header.frame_id
                            else "camera_color_optical_frame")

            # =========================================================
            # operator_only 模式：第一階段，生成操作臂夾取姿態
            # =========================================================
            if mode == "operator_only":
                rospy.loginfo("🧠 operator_only 模式：生成操作臂姿態...")

                op_mask_img = cv2.imread(operator_mask_path, cv2.IMREAD_GRAYSCALE)
                op_mask = cv2.resize(
                    op_mask_img, (color_np.shape[1], color_np.shape[0])) > 127
                pts_operator = np.stack(
                    [points_x, points_y, points_z],
                    axis=-1)[valid_depth_mask & op_mask].astype(np.float32)
                pts_operator = self.filter_outlier_points(pts_operator)
                lims_operator = self.get_dynamic_lims(pts_operator, pad=0.0)

                if not lims_operator:
                    rospy.logwarn("⚠️ 操作臂遮罩內缺乏有效點雲")
                    self.plan_pub.publish(json.dumps([]))
                    return

                gg_operator, _ = self.anygrasp.get_grasp(
                    points_full, colors_full,
                    lims=lims_operator,
                    apply_object_mask=False,
                    dense_grasp=True,
                    collision_detection=True)

                if gg_operator is None or len(gg_operator) == 0:
                    rospy.logwarn("⚠️ 操作臂遮罩內找不到姿態")
                    self.plan_pub.publish(json.dumps([]))
                    return

                top_k = 10
                gg_operator = gg_operator.nms().sort_by_score()[:top_k]
                # 傳入 ROS TF 名稱 "rightarm"（操作臂）
                operator_filtered = self.get_arm_specific_grasps(
                    gg_operator, "rightarm", camera_frame)

                if self.cfgs.debug:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points_full)
                    cloud.colors = o3d.utility.Vector3dVector(colors_full)
                    vis_list = [cloud]
                    for item in operator_filtered:
                        vis_list.append(item['grasp'].to_open3d_geometry())
                    threading.Thread(
                        target=lambda vl=vis_list: o3d.visualization.draw_geometries(vl),
                        daemon=True
                    ).start()

                json_plan = self.generate_operator_plan(operator_filtered, camera_frame)

                if json_plan:
                    self.plan_pub.publish(json.dumps(json_plan))
                    rospy.loginfo(
                        f"✅ operator_only 計畫書發布，共 {len(json_plan)} 個姿態")
                else:
                    rospy.logwarn("❌ operator_only 模式找不到有效姿態")
                    self.plan_pub.publish(json.dumps([]))
                return

            # =========================================================
            # receiver_only 模式：第二階段，生成接收臂接取姿態
            # =========================================================
            if mode == "receiver_only":
                recv_mask_img = cv2.imread(receiver_mask_path, cv2.IMREAD_GRAYSCALE)
                recv_mask = cv2.resize(
                    recv_mask_img, (color_np.shape[1], color_np.shape[0])) > 127
                pts_receiver = np.stack(
                    [points_x, points_y, points_z],
                    axis=-1)[valid_depth_mask & recv_mask].astype(np.float32)
                pts_receiver = self.filter_outlier_points(pts_receiver)
                lims_receiver = self.get_dynamic_lims(pts_receiver, pad=0.0)

                if not lims_receiver:
                    rospy.logwarn("⚠️ 接收臂遮罩內缺乏有效點雲")
                    return
                rospy.loginfo("🧠 receiver_only 模式：生成接收臂姿態...")
                gg_receiver, _ = self.anygrasp.get_grasp(
                    points_full, colors_full,
                    lims=lims_receiver,
                    apply_object_mask=False,
                    dense_grasp=True,
                    collision_detection=True)

                if gg_receiver is None or len(gg_receiver) == 0:
                    rospy.logwarn("⚠️ 接收臂遮罩內找不到姿態")
                    self.plan_pub.publish(json.dumps([]))
                    return

                top_k = 10
                gg_receiver = gg_receiver.nms().sort_by_score()[:top_k]
                # 傳入 ROS TF 名稱 "leftarm"（接收臂）
                receiver_filtered = self.get_arm_specific_grasps(
                    gg_receiver, "leftarm", camera_frame)

                if self.cfgs.debug:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(points_full)
                    cloud.colors = o3d.utility.Vector3dVector(colors_full)
                    vis_list = [cloud]
                    for item in receiver_filtered:
                        vis_list.append(item['grasp'].to_open3d_geometry())
                    threading.Thread(
                        target=lambda vl=vis_list: o3d.visualization.draw_geometries(vl),
                        daemon=True
                    ).start()

                json_plan = self.generate_receiver_plan(receiver_filtered, camera_frame)

                if json_plan:
                    self.plan_pub.publish(json.dumps(json_plan))
                    rospy.loginfo(
                        f"✅ receiver_only 計畫書發布，共 {len(json_plan)} 個姿態")
                else:
                    rospy.logwarn("❌ receiver_only 模式找不到有效姿態")
                    self.plan_pub.publish(json.dumps([]))
                return

        except Exception as e:
            rospy.logerr(f"管線發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            self.plan_pub.publish(json.dumps([]))


if __name__ == '__main__':
    node = AnyGraspHandoverNode()
    rospy.sleep(2)
    rospy.spin()
