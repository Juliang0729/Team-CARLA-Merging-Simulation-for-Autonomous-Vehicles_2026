#!/bin/bash
set -e

# --- Check that CARLA_HOME is set ---
if [ -z "$CARLA_HOME" ]; then
    echo "Error: Please set \$CARLA_HOME before running this script"
    return 0
fi

# --- Default CARLA version if not set ---
if [ -z "$CARLA_VERSION" ]; then
   CARLA_VERSION="0.9.16"
fi

# --- Find any CARLA distribution file (egg or wheel) ---
CARLA_DIST_DIR="${CARLA_HOME}/PythonAPI/carla/dist"
CARLA_FILE=$(ls "$CARLA_DIST_DIR"/carla-"${CARLA_VERSION}"-*linux_x86_64.* 2>/dev/null | head -n 1)

if [ ! -f "$CARLA_FILE" ]; then
    echo "Error: CARLA file for version ${CARLA_VERSION} not found in $CARLA_DIST_DIR"
    echo "Please ensure you built CARLA's PythonAPI for your current Python version."
    return 0
fi

CACHE="${PWD}/cache"
mkdir -p "$CACHE"

echo "Copying CARLA file to cache folder..."
cp "$CARLA_FILE" "$CACHE"

if [[ "$CARLA_FILE" == *.egg ]]; then
    echo "Detected .egg format — unpacking for editable install..."
    unzip -o "$CARLA_FILE" -d "${CACHE}/carla-${CARLA_VERSION}"
    cp "${PWD}/scripts/setup.py" "${CACHE}/carla-${CARLA_VERSION}/"
    pip install -e "${CACHE}/carla-${CARLA_VERSION}"
else
    echo "Detected .whl format — installing directly..."
    pip install "$CARLA_FILE"
fi

echo "Setting environment variables..."
export PYTHONPATH=$PYTHONPATH:${CARLA_HOME}/PythonAPI/carla:${CARLA_HOME}/PythonAPI/carla/dist

echo "Successful Setup for CARLA ${CARLA_VERSION}!"

