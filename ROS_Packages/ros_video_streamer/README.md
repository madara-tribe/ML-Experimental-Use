# USB camera(video) stream by ROS

By using micro device like Jetson nano, capture image and send these to remote Host PC.

and display and check Host PC browser.

<img width="892" alt="staracture" src="https://user-images.githubusercontent.com/48679574/169679502-6959b8d2-0277-4a76-b143-94898e0c798c.png">

# ENV and Version
## Host OS
- Mac OS
- ROS noetic

## Guset OS
- Jetson Nano
- ROS melodic

# How to use

```zsh
$ git clone https://github.com/madara-tribe/ros_video_streamer.git
cd ~/catkin_make
# run usb camera
roscore
cd ~/ros_video_streamer/roscam_capture
cd roslaunch roscam_capture usb_cam.launch 

# display host PC browser
$ cd ~/ros_video_streamer/roscam_capture
$ rosrun ros_video_stream ros_video_stream _port:=10000
```

```zsh
# access to browser
topic_name = /usb_cam/image_raw
http://{device_ip_adress}:10000/stream?topic={topic_name}

# check ip address
$ ifconfig
```



# References
- [Opencv-CookBook](http://opencv.jp/cookbook/opencv_io.html)
- [mjpeg_server](https://github.com/RobotWebTools/mjpeg_server)
- [Raspberry Piロボットに搭載されたカメラの映像を遠隔PCに配信する](https://meltingrabbit.com/blog/article/2018122601/)
