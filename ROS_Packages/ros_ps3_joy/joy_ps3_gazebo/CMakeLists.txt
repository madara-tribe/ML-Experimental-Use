cmake_minimum_required(VERSION 3.0.2)
project(joy_ps3_gazebo)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

find_package(gazebo REQUIRED)
catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp std_msgs sensor_msgs nav_msgs tf gazebo_ros
    DEPENDS GAZEBO
)



###########
## Build ##
###########


install(DIRECTORY launch models rviz worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


