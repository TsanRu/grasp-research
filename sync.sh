#!/bin/bash
# Sync from development workspace to this GitHub repo, then commit and push.
# Run from /home/rvl/grasp_research/

set -e

DEV=/home/rvl/ros_ws/src
REPO=/home/rvl/grasp_research/src

echo "==> Syncing ros_ur3..."
rsync -a \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='recordings' \
  --exclude='screenshots' \
  --exclude='my_gazebo_data' \
  --exclude='*.so' \
  --exclude='build' \
  --exclude='*.mp4' \
  --exclude='*.bag' \
  --exclude='ur_ikfast/build' \
  --exclude='ur_ikfast/*.so' \
  --exclude='ur_ikfast/*.dae' \
  --exclude='ur_ikfast/*.cpp' \
  --delete \
  "$DEV/ros_ur3/" "$REPO/ros_ur3/"

echo "==> Syncing anygrasp_ros scripts..."
rsync -a --include='*.py' --exclude='*' \
  "$DEV/anygrasp_sdk/grasp_detection/" "$REPO/anygrasp_ros/scripts/"

rsync -a \
  "$DEV/anygrasp_sdk/grasp_detection/lang-segment-anything/semantic_grasp_analyzer.py" \
  "$DEV/anygrasp_sdk/grasp_detection/lang-segment-anything/vision_pipeline.py" \
  "$REPO/anygrasp_ros/scripts/lang_segment_anything/"

echo "==> Syncing foundationpose_ros..."
rsync -a \
  "$DEV/FoundationPose/foundationpose_node.py" \
  "$REPO/foundationpose_ros/scripts/"

rsync -a \
  "$DEV/FoundationPose/docker/" \
  "$REPO/foundationpose_ros/docker/"

rsync -a \
  "$DEV/FoundationPose/install_foundationpose.txt" \
  "$REPO/foundationpose_ros/"

echo ""
echo "==> Changes:"
git -C /home/rvl/grasp_research diff --stat HEAD

echo ""
read -p "Commit message (leave empty to skip commit): " MSG
if [ -n "$MSG" ]; then
  git -C /home/rvl/grasp_research add -A
  git -C /home/rvl/grasp_research commit -m "$MSG"
  read -p "Push to GitHub? [y/N] " PUSH
  if [ "$PUSH" = "y" ] || [ "$PUSH" = "Y" ]; then
    git -C /home/rvl/grasp_research push
    echo "==> Pushed."
  fi
else
  echo "==> Skipped commit."
fi
