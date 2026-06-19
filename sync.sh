#!/bin/bash
# Sync from development workspace to this GitHub repo, then commit and push.
# Run from /home/rvl/grasp_research/
#
# ⚠️ DO NOT edit files directly inside grasp_research/src — this is a one-way
# sync (DEV -> REPO). Anything edited here will be silently overwritten the
# next time this script runs. Always edit in $DEV.

set -e

DEV=/home/rvl/ros_ws/src
REPO=/home/rvl/grasp_research/src

# ---------------------------------------------------------------------------
# Ensure target directories exist (rsync can fail silently/error if missing)
# ---------------------------------------------------------------------------
mkdir -p "$REPO/anygrasp_ros/scripts/lang_segment_anything"
mkdir -p "$REPO/foundationpose_ros/scripts"
mkdir -p "$REPO/foundationpose_ros/docker"

# ---------------------------------------------------------------------------
# 1. ros_ur3 — full mirror sync (safe to use --delete: this is a 1:1 mirror)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. anygrasp_ros — selective file sync (no --delete: target may hold other
#    hand-placed files like __init__.py, configs, etc. that shouldn't vanish)
# ---------------------------------------------------------------------------
echo "==> Syncing anygrasp_ros scripts..."
rsync -a --include='*.py' --exclude='*' \
  "$DEV/anygrasp_sdk/grasp_detection/" "$REPO/anygrasp_ros/scripts/"

rsync -a \
  "$DEV/anygrasp_sdk/grasp_detection/lang-segment-anything/semantic_grasp_analyzer.py" \
  "$DEV/anygrasp_sdk/grasp_detection/lang-segment-anything/vision_pipeline.py" \
  "$REPO/anygrasp_ros/scripts/lang_segment_anything/"

# ---------------------------------------------------------------------------
# 3. foundationpose_ros — selective file sync (no --delete, same reasoning)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4. Secret scan — second line of defense before committing
#    (catches API keys / tokens / passwords accidentally left in code)
# ---------------------------------------------------------------------------
echo ""
echo "==> Scanning for potential secrets..."
SECRET_HITS=$(git -C /home/rvl/grasp_research diff HEAD | grep -E -i "(api[_-]?key|secret|token|password|sk-[a-zA-Z0-9]{20,})" || true)

if [ -n "$SECRET_HITS" ]; then
  echo "⚠️  WARNING: Potential secret detected in the diff below:"
  echo "$SECRET_HITS"
  echo ""
  read -p "Continue anyway? [y/N] " CONFIRM
  if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "==> Aborted. Review and remove the secret before re-running."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 5. Review changes, commit, push
# ---------------------------------------------------------------------------
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
