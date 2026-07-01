sed -i 's/\r$//' scripts/ros_entrypoint.sh
sed -i 's/\r$//' scripts/ros2_setup_scripts_ubuntu.sh
chmod +x scripts/ros_entrypoint.sh
docker build -t camera_node:latest .
