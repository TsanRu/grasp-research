cmake_minimum_required(VERSION 3.0.2)
project(anygrasp_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/anygrasp_ros.py
  scripts/anygrasp_ros_node_merge.py
  scripts/anygrasp_ros_handineye.py
  scripts/brain.py
  scripts/brain_handineye.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
