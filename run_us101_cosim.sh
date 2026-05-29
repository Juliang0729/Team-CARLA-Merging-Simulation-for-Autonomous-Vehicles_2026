#!/bin/bash
# OpenCDA US101 Co-Simulation Launcher
# Starts CARLA packaged build and runs OpenCDA simulation

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CARLA_DIR="/home/julian/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor"
CARLA_EXEC="./CarlaUE4.sh"
OPENCDA_DIR="/home/julian/OpenCDA"
CARLA_PID=""
APPLY_ML=""

# Parse arguments
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch CARLA and run OpenCDA US101 co-simulation"
    echo ""
    echo "Options:"
    echo "  --evaluation        Enable state estimation + performance evaluation"
    echo "  --state_estimator   Enable only state estimation (no evaluation)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                   # Run basic simulation with ML frameworks"
    echo "  $0 --evaluation      # Run with state estimation + evaluation"
    echo "  $0 --state_estimator # Run with state estimation only"
    echo ""
    echo "Note: --apply_ml is always enabled for ML/DL framework support"
}

# Parse command line arguments
OPENCDA_ARGS="--apply_ml"  # Always enable ML/DL frameworks
for arg in "$@"; do
    case $arg in
        --evaluation)
            OPENCDA_ARGS="$OPENCDA_ARGS --state_estimator --evaluation"
            shift
            ;;
        --state_estimator)
            OPENCDA_ARGS="$OPENCDA_ARGS --state_estimator"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$CARLA_PID" ] && ps -p $CARLA_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping CARLA (PID: $CARLA_PID)...${NC}"
        kill $CARLA_PID
        wait $CARLA_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Check if CARLA directory exists
if [ ! -d "$CARLA_DIR" ]; then
    echo -e "${RED}Error: CARLA directory not found: $CARLA_DIR${NC}"
    exit 1
fi

# Check if OpenCDA directory exists
if [ ! -d "$OPENCDA_DIR" ]; then
    echo -e "${RED}Error: OpenCDA directory not found: $OPENCDA_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}=== OpenCDA US101 Co-Simulation Launcher ===${NC}"
echo ""

# Start CARLA
echo -e "${YELLOW}Starting CARLA from packaged build...${NC}"
cd "$CARLA_DIR"
$CARLA_EXEC &
CARLA_PID=$!
echo -e "${GREEN}CARLA started with GUI (PID: $CARLA_PID)${NC}"

# Wait for CARLA to be ready
echo -e "${YELLOW}Waiting for CARLA to initialize...${NC}"
sleep 15

# Check if CARLA is still running
if ! ps -p $CARLA_PID > /dev/null; then
    echo -e "${RED}Error: CARLA failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}CARLA is ready!${NC}"
echo ""

# Start OpenCDA simulation
echo -e "${YELLOW}Starting OpenCDA US101 co-simulation...${NC}"
cd "$OPENCDA_DIR"

# Activate conda environment if needed
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ] || [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate opencda
    echo -e "${GREEN}Activated conda environment: opencda${NC}"
fi

# Run OpenCDA
echo -e "${YELLOW}Running: python opencda.py -t single_us101_cosim $OPENCDA_ARGS${NC}"
echo ""
python opencda.py -t single_us101_cosim $OPENCDA_ARGS

echo -e "\n${GREEN}Simulation completed!${NC}"
