# Make robut move on ROS simulation by ps3 controller 

## Version
- Ubuntu 20.04
- ROS noetic

## Setup
```zsh
# install teleop-twist
sudo apt-get install ros-noetic-joy
sudo apt install ros-noetic-teleop-twist-joy

# for turtlebot3 world
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3.git
git clone -b noetic-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

cd ~/catkin_ws && catkin_make
source ~/.bashrc
```

# PS3 controller and button
If ps3 controller is connected, check by this command
```zsh
$ ls -l /dev/input/js*
### crw-rw-r--+ 1 root input 13, 0  7 13:53 /dev/input/js0
```
<img src="https://user-images.githubusercontent.com/48679574/157031297-c4567f7e-a34d-41c8-96e0-379c7db03609.jpg" width="270px"><img src="https://user-images.githubusercontent.com/48679574/157031307-7d2ce401-13d0-4443-b7cd-793df5a549e8.jpg" width="270px"><img src="https://user-images.githubusercontent.com/48679574/157031310-ef4a8f19-431d-44e5-b156-8f7663c93d63.jpg" width="270px">



## PS3 controller movement check.
```zsh
# After push PS button
$ roscore
$ rosrun joy joy_node
### [ WARN] [1646630772.708410626]: Couldn't set gain on joystick force feedback: Bad file descriptor
### [ INFO] [1646630772.710396915]: Opened joystick: /dev/input/js0 (Sony PLAYSTATION(R)3 Controller). deadzone_: 0.050000

$rostopic echo /joy
```


## requirements package
```zsh
sudo apt-get install ros-noetic-joy ros-noetic-teleop-twist-joy
#sudo apt-get install ros-noetic-joystick-drivers
```


# Performance

## control turtlebot
with "ps3_joy_basic", you can control turtlebot by ps3 controller
```zsh
$ roslaunch ps3_joy_basic turtle.launch
```

<img src="https://user-images.githubusercontent.com/48679574/157000034-c1490365-5c96-42ce-af54-d113b6c5c5be.png" width="500px">


## control on rviz
with "ps3_rviz_move", you can control custmize your robut on Rviz by ps3 controller
```zsh
$ roslaunch ps3_rviz_move movement_robot.launch joy:=true
```

<img src="https://user-images.githubusercontent.com/48679574/157000012-3bf8fec2-4033-489b-a876-524f8819d67e.png" width="500px">


## control on gazebo
with pushing 0-buttun(selsect button), use 1-button for rotate and use 2-button for move ahead or back

```zsh
$ roslaunch joy_ps3_gazebo wheel_gazebo.launch
$ roslaunch joy_ps3_gazebo joy_control.launch
```

<img src="https://user-images.githubusercontent.com/48679574/157031310-ef4a8f19-431d-44e5-b156-8f7663c93d63.jpg" width="300px"><img src="https://user-images.githubusercontent.com/48679574/160335697-75a8d1be-015a-417c-bcd0-efabef3eeec2.png" width="400px">

# References
・[ros_lecture](https://github.com/project-srs/ros_lecture)

・[ROS.org-ps3joy](http://wiki.ros.org/ps3joy)

・[ROS講座10 カスタムROSメッセージ](https://qiita.com/srs/items/7ac023c549e585caeed0)
