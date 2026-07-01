#!/bin/bash
# setup_udev.sh - Host-side udev setup for Logitech Brio 100
#
# What this does:
#   1. Installs udev rule  → creates stable symlink /dev/video/brio100
#   2. Installs systemd service → auto-sets resolution/fps on plug-in
#   3. Adds current user to the 'video' group
#
# Run once on the host machine before starting Docker.
# After running, unplug and replug the camera (or run the trigger command below).
#
# To reset / uninstall: run with --reset flag
#   sudo ./setup_udev.sh --reset

set -e

RULES_FILE="99-brio100.rules"
SERVICE_FILE="brio100-setup@.service"
UDEV_RULES_DIR="/etc/udev/rules.d"
SYSTEMD_DIR="/etc/systemd/system"
CURRENT_USER="${SUDO_USER:-$(whoami)}"

# ── Reset mode ────────────────────────────────────────────────────
if [[ "${1:-}" == "--reset" ]]; then
  echo "[reset] Removing udev rule and systemd service..."
  sudo rm -f "${UDEV_RULES_DIR}/${RULES_FILE}"
  sudo rm -f "${SYSTEMD_DIR}/${SERVICE_FILE}"
  sudo udevadm control --reload-rules
  sudo systemctl daemon-reload
  sudo udevadm trigger --subsystem-match=video4linux
  sudo rm -f /dev/video/brio100
  echo "[reset] Done. Symlink and rules removed."
  exit 0
fi

# ── Require root ──────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Please run with sudo: sudo ./setup_udev.sh"
  exit 1
fi

# ── Check source files exist ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/${RULES_FILE}" ]]; then
  echo "ERROR: ${RULES_FILE} not found in ${SCRIPT_DIR}"
  exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/${SERVICE_FILE}" ]]; then
  echo "ERROR: ${SERVICE_FILE} not found in ${SCRIPT_DIR}"
  exit 1
fi

# ── Install udev rule ─────────────────────────────────────────────
echo "[1/4] Installing udev rule: ${RULES_FILE}"
cp "${SCRIPT_DIR}/${RULES_FILE}" "${UDEV_RULES_DIR}/${RULES_FILE}"

# ── Install systemd service ───────────────────────────────────────
echo "[2/4] Installing systemd service: ${SERVICE_FILE}"
cp "${SCRIPT_DIR}/${SERVICE_FILE}" "${SYSTEMD_DIR}/${SERVICE_FILE}"

# ── Add user to video group ───────────────────────────────────────
echo "[3/4] Adding user '${CURRENT_USER}' to video group"
usermod -aG video "${CURRENT_USER}"

# ── Reload rules and trigger ──────────────────────────────────────
echo "[4/4] Reloading udev rules and triggering..."
udevadm control --reload-rules
systemctl daemon-reload
udevadm trigger --subsystem-match=video4linux

# ── Verify ────────────────────────────────────────────────────────
echo ""
echo "Setup complete. Verifying symlink..."
sleep 1

if [[ -L /dev/video/brio100 ]]; then
  echo "  [OK] /dev/video/brio100 -> $(readlink /dev/video/brio100)"
else
  echo "  [WARN] Symlink /dev/video/brio100 not found yet."
  echo "         Try unplugging and replugging the camera."
fi

echo ""
echo "Test with:"
echo "  v4l2-ctl -d /dev/video/brio100 --get-fmt-video"
echo ""
echo "NOTE: Log out and back in (or run 'newgrp video') for group change to take effect."
