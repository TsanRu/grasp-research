# Robot Grasp Research Workspace

A ROS Noetic workspace for dual-arm UR3 robot grasping research, integrating AnyGrasp grasp detection and FoundationPose 6-DoF object pose estimation.

## System Overview

```
┌─────────────────────────────────────────────────┐
│  Gazebo Simulation (ros_ur3)                    │
│  Dual UR3 + RealSense D435 + Robotiq gripper   │
└────────────────────┬────────────────────────────┘
                     │ ROS topics
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌─────────────────┐
│ anygrasp_ros │ │ YOLO-3D  │ │foundationpose_ros│
│ Grasp pose   │ │ Object   │ │ 6-DoF pose est. │
│ generation   │ │ detect.  │ │ (Docker-based)  │
└──────┬───────┘ └────┬─────┘ └────────┬────────┘
       └──────────────┴────────────────┘
                       │
              ┌────────▼────────┐
              │  ur_control     │
              │  Arm planning   │
              │  & execution    │
              └─────────────────┘
```

## Repository Structure

```
src/
├── ros_ur3/              # Dual UR3 simulation environment (fork of cambel/ur3)
│   ├── ur_control/       # Robot control library + research scripts
│   ├── ur_gripper_gazebo/# Gazebo world, models, launch files
│   ├── ur_gripper_85_moveit_config/
│   ├── ur_hande_moveit_config/
│   ├── ur_gripper_description/
│   ├── ur_handeye_calibration/
│   └── ur_pykdl/
├── anygrasp_ros/         # AnyGrasp ROS integration (this repo's custom code)
│   └── scripts/
│       ├── brain.py              # Main grasp pipeline controller
│       ├── brain_handineye.py    # Hand-in-eye variant
│       ├── anygrasp_ros.py       # AnyGrasp ROS node
│       ├── grasppose_generating.py
│       └── lang_segment_anything/# Language-guided segmentation integration
└── foundationpose_ros/   # FoundationPose ROS bridge (Docker-based)
    ├── scripts/
    │   └── foundationpose_node.py
    └── docker/           # Docker setup for FoundationPose
```

## Dependencies

### Required ROS Packages
```bash
sudo apt install ros-noetic-moveit ros-noetic-ur-robot-driver \
  ros-noetic-realsense2-camera ros-noetic-gazebo-ros-pkgs
```

### Third-party packages (clone into `src/`)
```bash
cd src/

# AnyGrasp SDK (requires license from Graspnet)
# https://github.com/graspnet/anygrasp_sdk
# Place in src/anygrasp_sdk/ — the anygrasp_ros package above depends on it

# FoundationPose (run via Docker, see foundationpose_ros/docker/)
# https://github.com/NVlabs/FoundationPose

# UR ROS Driver
git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver

# RealSense ROS
git clone https://github.com/IntelRealSense/realsense-ros -b ros1-legacy
```

### ur_ikfast (IK solver)
```bash
pip install ur-ikfast
```

### Python dependencies
```bash
pip install numpy scipy open3d
# For anygrasp_ros:
pip install graspnetAPI
# For lang-segment-anything:
pip install lang-segment-anything
```

## Build

```bash
cd ~/grasp_research
catkin_make
source devel/setup.bash
```

## FoundationPose (Docker)

FoundationPose runs inside Docker and communicates with ROS via the `foundationpose_node.py` bridge.

```bash
cd src/foundationpose_ros/docker
bash run_container.sh
```

See `src/foundationpose_ros/install_foundationpose.txt` for detailed setup.

## Usage

### Launch simulation
```bash
roslaunch ur_gripper_gazebo ur3_dual_bringup.launch
```

### Run grasp pipeline
```bash
# Terminal 1: AnyGrasp node
rosrun anygrasp_ros anygrasp_ros.py

# Terminal 2: Brain (pipeline controller)
rosrun anygrasp_ros brain.py

# Terminal 3: FoundationPose node (requires Docker running)
rosrun foundationpose_ros foundationpose_node.py
```

## Based On

- [cambel/ur3](https://github.com/cambel/ur3) — UR3 dual-arm simulation framework
- [graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk) — AnyGrasp grasp detection
- [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose) — 6-DoF pose estimation
- [luca-medeiros/lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything) — Language-guided segmentation
