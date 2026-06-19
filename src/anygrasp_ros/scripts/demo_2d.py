import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

# --- 原有的參數設定，無需修改 ---
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='模型檢查點路徑')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='夾爪最大寬度 (<=0.1公尺)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='夾爪高度')
parser.add_argument('--top_down_grasp', action='store_true', help='僅輸出由上往下的抓取')
parser.add_argument('--debug', action='store_true', help='啟用除錯模式')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # --- 步驟 1: 讀取影像與相機參數 (無變動) ---
    color_path = os.path.join(data_dir, 'saved_color.png')
    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth_path = os.path.join(data_dir, 'saved_depth.npy')
    depths = np.load(depth_path)
    
    fx, fy = 462.1675733606097, 462.1675733606097
    cx, cy = 320.5, 240.5
    scale = 1.0
    height, width = depths.shape

    # --- 先產生完整的場景點雲 (與之前相同) ---
    xmap, ymap = np.arange(width), np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    
    full_points = np.stack([points_x, points_y, points_z], axis=-1)
    
    # ==================== 全新步驟 2: 使用 2D 影像邊界框來過濾點雲 ====================
    # 定義一個2D邊界框 [xmin, ymin, xmax, ymax] 單位是像素(pixel)。
    # [左上角的x, 左上角的y, 右下角的x, 右下角的y]
    # 您需要根據您的影像來估計並微調這些數值。
    bbox_2d = [150, 200, 600, 450]  # 這只是範例值，請務必親自微調！
    
    xmin, ymin, xmax, ymax = bbox_2d

    # 使用Numpy的陣列切片(slicing)功能，直接取出2D框選區域內的所有資料
    points_in_bbox = full_points[ymin:ymax, xmin:xmax]
    colors_in_bbox = colors[ymin:ymax, xmin:xmax]
    depths_in_bbox = depths[ymin:ymax, xmin:xmax]

    # 過濾掉深度值為0或無效的點
    valid_depth_mask = (depths_in_bbox > 0)
    
    # 將2D區域內所有有效的點轉換成 (N, 3) 的點雲陣列
    points_filtered = points_in_bbox[valid_depth_mask].astype(np.float32)
    colors_filtered = colors_in_bbox[valid_depth_mask].astype(np.float32)

    if len(points_filtered) == 0:
        print("警告：您設定的2D邊界框內沒有任何有效的深度點，請調整 bbox_2d 的數值。")
        return
        
    print(f"在2D邊界框內共找到 {len(points_filtered)} 個有效點。")

    # ==================== 步驟 3: 在過濾後的點雲上執行抓取偵測 (無變動) ====================
    lims = [-1, 1, -1, 1, 0, 1.5]
    gg, cloud = anygrasp.get_grasp(points_filtered, colors_filtered, lims=lims)

    if len(gg) == 0:
        print('在此區域沒有偵測到任何有效的抓取姿態！')
        return

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(f"偵測到 {len(gg)} 個抓取, 最高分: {gg_pick[0].score:.4f}")

    # --- 步驟 4: 視覺化結果 (無變動) ---
    if cfgs.debug:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_filtered)
        pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

        trans_mat = np.array([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])
        pcd.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        
        print("顯示在2D邊界框內偵測到的所有抓取姿態...")
        o3d.visualization.draw_geometries([*grippers, pcd])
        
        print("顯示分數最高的抓取姿態...")
        o3d.visualization.draw_geometries([grippers[0], pcd])

if __name__ == '__main__':
    demo('.')
