import numpy as np

# 定義可樂罐的中心點 (X, Y, Z)
coke_can_center = np.array([0.9, 0.3, 0.85])

# 設定不同的夾取角度
grasp_angles = np.linspace(0, np.pi, 6)  # 產生 6 種不同的夾取方向

# 計算可能的夾取位姿
grasp_poses = []
for angle in grasp_angles:
    x_offset = 0.05 * np.cos(angle)
    y_offset = 0.05 * np.sin(angle)
    
    grasp_pose = {
        "position": coke_can_center + np.array([x_offset, y_offset, 0]),
        "orientation": angle  # 假設是 2D 旋轉
    }
    grasp_poses.append(grasp_pose)

print("可能的夾取位姿:", grasp_poses)
