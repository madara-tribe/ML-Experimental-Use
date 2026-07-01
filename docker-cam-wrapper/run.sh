# ビルド
cd final_ws
colcon build

source install/setup.bash

# Python版を起動
ros2 launch camera_bringup camera_bringup.launch.py backend:=python

# C++版を起動
ros2 launch camera_bringup camera_bringup.launch.py backend:=cpp

# デバイスパスを変える場合
ros2 launch camera_bringup camera_bringup.launch.py backend:=cpp device_path:=/dev/video2
```

---

## ワークスペース構成
```
final_ws/src/
├── camera_publisher/        ← Python publisher
├── camera_publisher_cpp/    ← C++ publisher
├── camera_subscriber/       ← Python subscriber
├── camera_subscriber_cpp/   ← C++ subscriber
└── camera_bringup/          ← 統合launchファイル（backend引数で切替）
