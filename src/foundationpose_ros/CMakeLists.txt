cmake_minimum_required(VERSION 3.0.2)
project(foundationpose_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/foundationpose_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
