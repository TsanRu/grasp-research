import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    
    # === 關鍵修改：已更新為你的相機參數 ===
    # 從你的 rostopic echo /camera/color/camera_info 取得
    # K: [462.16757..., 0.0, 320.5, 0.0, 462.16757..., 240.5, 0.0, 0.0, 1.0]
    fx, fy = 462.1675733606097, 462.1675733606097
    cx, cy = 320.5, 240.5
    scale = 1000.0
    
    # === 👉 下一步行動：調整你的工作空間 (lims) ===
    # 這是提升抓取分數最重要的一步！
    # 你需要根據 Gazebo 場景估算物體的 (X, Y, Z) 範圍，把手臂和背景排除掉。
    # 座標系: +X 向右, +Y 向下, +Z 向前
    #
    # 範例 (請務必根據你的場景修改這些值！):
    xmin, xmax = -0.2, 0.2   # 框住最左到最右的物體
    ymin, ymax = -0.2, 0.2  # 框住物體的高度範圍
    zmin, zmax = 0.4, 0.9    # 設定物體離相機的距離範圍
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    # 注意：這裡的 mask 只是初步過濾，真正的 workspace 限制是由 lims 參數在 get_grasp 函式中完成的
    mask = (points_z > 0) & (points_z < 1.5) # 將 Z 軸過濾範圍稍微放大，避免切掉物體
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print("Point cloud min/max before filtering:")
    print(points.min(axis=0), points.max(axis=0))

    # 執行抓取偵測，lims 參數會在這裡過濾點雲
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return # 如果沒有抓取點，直接結束函式避免後續出錯

    gg = gg.nms().sort_by_score()
    # gg_pick = gg[0:20]
    # print("Detected grasp scores:")
    # print(gg_pick.scores)
    # print('Best grasp score:', gg_pick[0].score)

    # # visualization
    # if cfgs.debug:
    #     trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    #     cloud.transform(trans_mat)
    #     grippers = gg.to_open3d_geometry_list()
    #     for gripper in grippers:
    #         gripper.transform(trans_mat)
    #     print("Showing all detected grasps in 3D...")
    #     o3d.visualization.draw_geometries([*grippers, cloud])
    #     print("Showing the best grasp in 3D...")
    #     o3d.visualization.draw_geometries([grippers[0], cloud])
    
    
    # === ⭐️ 新增部分：提取並打印前 5 個抓取的 6-DoF 數據 ===
    # 導入用於旋轉變換的函式庫
    from scipy.spatial.transform import Rotation

    # 我們只看分數最高的前 5 個候選抓取
    top_k_grasps = 5
    gg_pick = gg[0:top_k_grasps]

    print("\n" + "="*40)
    print(f"Displaying Top {len(gg_pick)} Grasp Candidates:")
    print("="*40)

    for i, grasp in enumerate(gg_pick):
        print(f"--- Grasp Candidate #{i} ---")
        
        # 1. 抓取分數 (Score) 和夾爪寬度 (Width)
        print(f"Score: {grasp.score:.4f}")
        print(f"Gripper Width: {grasp.width:.4f} (m)")

        # 2. 位置 (Translation - X, Y, Z)
        #    這是抓取點相對於「相機座標系」的位置
        translation = grasp.translation
        print(f"Position (X, Y, Z): [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")

        # 3. 姿態 (Orientation - 3x3 Rotation Matrix)
        rotation_matrix = grasp.rotation_matrix
        print("Orientation (Rotation Matrix):")
        print(np.round(rotation_matrix, 3))

        # 4. 將旋轉矩陣轉換為四元數 (Quaternion)
        #    這是在 ROS 中更常用的格式
        try:
            r = Rotation.from_matrix(rotation_matrix)
            quaternion = r.as_quat() # 格式為 [x, y, z, w]
            print(f"Orientation (Quaternion x,y,z,w): [{quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f}]")
        except Exception as e:
            print(f"Could not convert to quaternion: {e}")
            
        print("-" * 20)

    # --- 視覺化部分 ---
    if cfgs.debug:
        # 取得所有抓取的 Open3D 幾何模型
        grippers = gg.to_open3d_geometry_list()
        
        # 座標系轉換，讓 Z 軸朝上，方便觀察
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        for gripper in grippers:
            gripper.transform(trans_mat)

        # === ⭐️ 對應方法：一次只看一個抓取 ===
        # 打印出來的 Grasp #0 對應的就是 grippers[0]
        # 打印出來的 Grasp #1 對應的就是 grippers[1]，以此類推
        
        # 顯示分數最高的抓取 (Grasp #0)
        print("Showing the BEST grasp (#0) in 3D...")
        o3d.visualization.draw_geometries([grippers[0], cloud])
        
        # (可選) 如果你想看第二好的抓取 (Grasp #1)
        # print("Showing the 2nd best grasp (#1) in 3D...")
        # o3d.visualization.draw_geometries([grippers[1], cloud])

        # (可選) 顯示所有抓取
        # print("Showing all detected grasps in 3D...")
        # o3d.visualization.draw_geometries([*grippers, cloud])


if __name__ == '__main__':
    demo('./my_gazebo_data/')