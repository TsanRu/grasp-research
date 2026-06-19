import sys
import copy
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import numpy as np
import pandas as pd

def scan_reachability(move_group, x_range, y_range, z_range, step, orientation_rpy, output_csv):
    results = []
    roll, pitch, yaw = orientation_rpy
    quat = quaternion_from_euler(roll, pitch, yaw)
    for x in np.arange(x_range[0], x_range[1], step):
        for y in np.arange(y_range[0], y_range[1], step):
            for z in np.arange(z_range[0], z_range[1], step):
                pose = Pose()
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                move_group.set_pose_target(pose)
                waypoints = [copy.deepcopy(pose)]
                (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, True)
                reachable = (fraction > 0.95) and (len(plan.joint_trajectory.points) > 0)
                results.append({'x': x, 'y': y, 'z': z, 'reachable': bool(reachable)})
                print(f"({x:.2f}, {y:.2f}, {z:.2f}) -> {'O' if reachable else 'X'}")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"掃描完成，結果已存至 {output_csv}")
    return df

def find_both_arms_reachable(left_csv, right_csv, output_csv='both_arms_reachable.csv'):
    left_df = pd.read_csv(left_csv)
    right_df = pd.read_csv(right_csv)
    left_reach = left_df[left_df['reachable'] == True]
    right_reach = right_df[right_df['reachable'] == True]
    left_coords = set(zip(left_reach['x'], left_reach['y'], left_reach['z']))
    right_coords = set(zip(right_reach['x'], right_reach['y'], right_reach['z']))
    both_reachable = left_coords & right_coords
    both_df = pd.DataFrame(list(both_reachable), columns=['x', 'y', 'z'])
    both_df.to_csv(output_csv, index=False)
    print(f"雙臂同時可達點數量：{both_df.shape[0]}")
    print(f"已存至 {output_csv}")
    # print(both_df.head(10))
    return both_df

if __name__ == "__main__":
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('reachability_scan', anonymous=True)
    # 掃描參數
    x_range = (0.4, 1.15)
    y_range = (-0.2, 0.2)
    z_range = (0.87, 1.1)  # 0.7 + 0.17
    step = 0.03
    orientation_rpy = (0, 3.14159, 0)

    # 掃描 leftarm
    left_group = moveit_commander.MoveGroupCommander("leftarm")
    left_csv = "reachability_map_leftarm.csv"
    # scan_reachability(left_group, x_range, y_range, z_range, step, orientation_rpy, left_csv)

    # 掃描 rightarm
    right_group = moveit_commander.MoveGroupCommander("rightarm")
    right_csv = "reachability_map_rightarm.csv"
    # scan_reachability(right_group, x_range, y_range, z_range, step, orientation_rpy, right_csv)

    # 尋找雙臂同時可達點
    find_both_arms_reachable(left_csv, right_csv)

    moveit_commander.roscpp_shutdown()
