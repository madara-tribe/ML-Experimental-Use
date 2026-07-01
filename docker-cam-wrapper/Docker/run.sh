#!/bin/bash
# run.sh - Launch the camera Docker container
# Requires udev symlink /dev/video/brio100 to be set up on the host.
# Run Docker/setup_udev/setup_udev.sh first if not already done.

set -e

# ── Device configuration ──────────────────────────────────────────
# Primary: use stable udev symlink created by 99-brio100.rules
# Fallback: set BRIO_DEV env var to override (e.g. BRIO_DEV=/dev/video2 ./run.sh)
BRIO_DEV="${BRIO_DEV:-/dev/video/brio100}"
CONTAINER_DEV="/dev/video/brio100"

IMAGE_NAME="${IMAGE_NAME:-camera_node:latest}"
ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"

# ── Additional mount path ─────────────────────────────────────────
PLACE_HOST="${PLACE_HOST:-/home/hagi/Downloads/place}"
PLACE_CONTAINER="${PLACE_CONTAINER:-/home/hagi/Downloads/place}"

# ── Pre-flight checks ─────────────────────────────────────────────
if [[ ! -e "$BRIO_DEV" ]]; then
  echo "ERROR: Camera device '$BRIO_DEV' not found on host."
  echo ""
  echo "  If udev is not set up yet, run:"
  echo "    cd Docker/setup_udev && sudo ./setup_udev.sh"
  echo ""
  echo "  Or specify a raw device path manually:"
  echo "    BRIO_DEV=/dev/video2 ./run.sh"
  exit 1
fi

if [[ ! -d "$PLACE_HOST" ]]; then
  echo "ERROR: Mount directory '$PLACE_HOST' not found on host."
  exit 1
fi

echo "Using camera device: $BRIO_DEV -> $CONTAINER_DEV"
echo "Mounting ROS2 workspace: $ROS2_WS -> /ros2_ws"
echo "Mounting additional path: $PLACE_HOST -> $PLACE_CONTAINER"

# ── X11 forwarding (for GUI tools if needed) ──────────────────────
if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local:docker 2>/dev/null || true
  X11_ARGS=(
    --env="DISPLAY=$DISPLAY"
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
  )
else
  X11_ARGS=()
fi

# ── Launch container ──────────────────────────────────────────────
docker run -it --rm \
  --net=host \
  --privileged \
  "${X11_ARGS[@]}" \
  --device="${BRIO_DEV}:${CONTAINER_DEV}" \
  --group-add video \
  --volume="${ROS2_WS}:/ros2_ws" \
  --volume="${PLACE_HOST}:${PLACE_CONTAINER}" \
  "${IMAGE_NAME}"
