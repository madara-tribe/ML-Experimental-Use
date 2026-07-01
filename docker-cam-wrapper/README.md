# docker-camera-wrapper

Use a USB camera inside Docker and publish/subscribe images through ROS2.

This repository provides a ROS2 pub/sub camera system running inside Docker. Both Python and C++ implementations are included and can be switched with a single launch argument.

---

## Architecture

```
Host Machine
  └── USB Camera (/dev/video0)
        │  (udev rules → stable device path)
        ▼
  Docker Container
    └── ROS2 Workspace
          └── src/
                ├── camera_publisher      (Python)  ┐
                ├── camera_publisher_cpp  (C++)     ├─ publishes  /image_raw
                ├── camera_subscriber     (Python)  ┐
                ├── camera_subscriber_cpp (C++)     ├─ subscribes /image_raw
                └── camera_bringup                  └─ launch (backend:=python|cpp)
```

---

## Package Structure

```
final_ws/src/
├── camera_publisher/          # Python publisher node
│   └── camera_publisher/
│       └── camera_pub.py
├── camera_publisher_cpp/      # C++ publisher node
│   └── src/
│       └── camera_pub.cpp
├── camera_subscriber/         # Python subscriber node
│   └── camera_subscriber/
│       └── camera_sub.py
├── camera_subscriber_cpp/     # C++ subscriber node
│   └── src/
│       └── camera_sub.cpp
└── camera_bringup/            # Unified launch package
    └── launch/
        └── camera_bringup.launch.py
```

| Package | Language | Role |
|---|---|---|
| `camera_publisher` | Python | Captures frames from USB camera, publishes to `/image_raw` |
| `camera_publisher_cpp` | C++ | Same as above |
| `camera_subscriber` | Python | Subscribes `/image_raw`, logs received frame size |
| `camera_subscriber_cpp` | C++ | Same as above |
| `camera_bringup` | Python (launch) | Launches pub+sub pair by `backend` argument |

---

## Setup

### 1. Configure udev (host machine)

Set up a udev rule so the camera always appears at a stable `/dev/` path.

```bash
# Check your camera attributes
udevadm info --name=/dev/video0 --attribute-walk

# Copy the udev rule
sudo cp Docker/setup_udev/99-realsense-libusb.rules /etc/udev/rules.d/

# Reload rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Verify (should show /dev/video0 with video group)
ls -l /dev/video0
```

### 2. Start the Docker container

```bash
cd Docker
./run.sh
```

The run script mounts the camera device into the container automatically.

### 3. Build the ROS2 workspace (inside Docker)

```bash
cd /ros2_ws
colcon build
source install/setup.bash
```

---

## Usage

### Launch with Python nodes (default)

```bash
ros2 launch camera_bringup camera_bringup.launch.py backend:=python
```

### Launch with C++ nodes

```bash
ros2 launch camera_bringup camera_bringup.launch.py backend:=cpp
```

### Specify a different camera device path

```bash
ros2 launch camera_bringup camera_bringup.launch.py backend:=python device_path:=/dev/video2
```

### Launch arguments

| Argument | Default | Description |
|---|---|---|
| `backend` | `python` | Node implementation to use: `python` or `cpp` |
| `device_path` | `/dev/video0` | Camera device path on the host |

---

## Node Details

### Publisher (`camera_pub`)

- Opens the camera via OpenCV (`cv::VideoCapture`)
- Resolution: 640×480, 30 fps
- Publishes `sensor_msgs/Image` to `/image_raw`
- QoS: `SensorDataQoS` (best effort, for real-time image transport)
- Uses `create_wall_timer` (C++) / `create_timer` (Python) — non-blocking

### Subscriber (`camera_sub`)

- Subscribes to `/image_raw` with matching `SensorDataQoS`
- Converts image message to OpenCV `Mat` via `cv_bridge`
- Logs received frame resolution to confirm the pipeline works

---

## Topic

| Topic | Type | Direction |
|---|---|---|
| `/image_raw` | `sensor_msgs/msg/Image` | publisher → subscriber |

---

## Dependencies

- ROS2 (Humble or later)
- OpenCV
- cv_bridge
- sensor_msgs
- rclpy (Python nodes)
- rclcpp (C++ nodes)
