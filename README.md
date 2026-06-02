## CARLA x SUMO Co-Simulation for Autonomous Vehicle Highway Merging Research
This repository is home to the CARLA-SUMO co-simulation developed by GWU MAE undergraduate students Abdu Eltahir, 
Connor Cheung, Joshua Yao, and Julian Gross as a part of their Senior Design Capstone project completed between
September 2025 and May 2026.

The project was adapted from the work of GWU PhD candidate Amin Tabrizian's study: [Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay](https://bpb-us-w2.wpmucdn.com/web.seas.gwu.edu/dist/9/15/files/2024/03/highway-merging-amin-9c2588ffcc670827.pdf) (co-authors: Zhitong Huang, Peng Wei). The merging controller logic is also sourced from Amin's [DAROM: Delay-Aware Reinforcement Learning for Highway On Ramp Merging](https://github.com/amin-tabrizian/onRampMerging) included here as a Git submodule.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install CARLA 0.9.16](#2-install-carla-0916)
  - [3. Set Up the Conda Environment](#3-set-up-the-conda-environment)
  - [4. Install the CARLA Python API](#4-install-the-carla-python-api)
  - [5. Install OpenCDA](#5-install-opencda)
- [Map Installation](#map-installation)
  - [Final Highway Model (FHWM)](#final-highway-model-fhwm)
  - [US Highway 101 (US101)](#us-highway-101-us101)
- [Running the Simulation](#running-the-simulation)
  - [US101 Co-Simulation (Recommended)](#us101-co-simulation-recommended)
  - [FHWM Co-Simulation](#fhwm-co-simulation)
  - [Command-Line Options](#command-line-options)
- [Optional Modules](#optional-modules)
  - [State Estimator](#state-estimator)
  - [Performance Evaluator](#performance-evaluator)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements a real-time co-simulation bridge between CARLA and SUMO to simulate a highway on-ramp merging scenario in which an autonomous ego vehicle is controlled by a pretrained **DAROM** (Delay-Aware Reinforcement learning for On-ramp Merging) Soft Actor-Critic agent. SUMO handles spawning and routing all vehicles, including the ego, while CARLA renders the scene in 3D. The bridge synchronizes the two simulators at every tick via the SUMO TraCI protocol.

Two custom maps are supported by the co-simulation:

| Map | Description |
|---|---|
| **FHWM** | A lightweight, legacy version of **US101** modeled after the U.S. 101 highway interchange with Lankershim Blvd. in Los Angeles, California. Available for both Linux and Windows systems. |
| **US101** | Updated version of **FHWM** complete with additional surrounding buildings and bug-fixes to road markings. Compiled into a Linux-only packaged CARLA build. |

---

## Architecture

The figure below depicts a basic overview of the co-simulation architecture.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4d4b7fe-19b6-4416-b492-5fb1e239a8b7" width="65%" />
</p>
<p align="center"><em>Figure 1: CARLA-SUMO Co-Simulation Architecture.</em></p>

A key aspect of the co-simulation bridge is that all vehicle commands are handled between SUMO and the merging controller. CARLA acts simply as a visualizer, never influencing the behavior of actors within SUMO. This distinction is crucial in ensuring 2D vehicle states from SUMO are properly reflected in CARLA's 3D space.

---

## Prerequisites

| Requirement | Version | Notes |
|:---:|:---:|:---:|
| **OS** | Ubuntu 20.04 / 22.04 | OpenCDA Co-simulation is exlusive to Linux systems |
| **Python** | 3.10.xx | Via Conda (see below) |
| **CARLA** | 0.9.16 | See installation steps |
| **SUMO** | ≥ 1.18 | `sudo add-apt-repository ppa:sumo/stable`<br>`sudo apt-get update`<br>`sudo apt-get install sumo sumo-tools sumo-doc` |
| **Conda** | Any recent version | Miniconda or Anaconda |
| **GPU** | Recommended | ≥ 8 GB VRAM |
| **Disk Space** | ≥ 40 GB | ≥ 40 GB of space for CARLA, SUMO, OpenCDA, Maps, Python Packages | 

> **Windows note:** The FHWM map assets can be extracted directly to `$CARLA_HOME` on Windows systems for standalone use (see [Final Highway Model (FHWM)](#final-highway-model-fhwm)).

---

## Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/Juliang0729/Team-CARLA-Merging-Simulation-for-Autonomous-Vehicles_2026.git OpenCDA
cd OpenCDA
```

> If you already cloned without `--recurse-submodules`, initialize the DAROM submodule manually:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Install CARLA 0.9.16

There are two versions of CARLA available for Linux systems: A prebuilt (packaged) version of the simulator and a version built from source. Both versions work with the co-simulation. The packaged version is recommended for the quickest and most lightweight installation process. Building CARLA from source is required if the user seeks to create their own maps in the future.

- **Package Installation:** Download the prebuilt CARLA 0.9.16 Linux binary from the [CARLA Releases page](https://github.com/carla-simulator/carla/releases/tag/0.9.16) (`CARLA_0.9.16.tar.gz`) and extract it to a directory of your choice (referred to as `$CARLA_HOME` below):
  
  ```bash
  tar -xzf CARLA_0.9.16.tar.gz -C ~/carla
  export CARLA_HOME=~/carla
  ```

- **Source Build:** Follow the instructions found in the [CARLA Documentation](https://carla.readthedocs.io/en/latest/build_linux/). The installation process takes ~4 hours and requires 100 GB of additional disk space.

### 3. Set Up the Conda Environment

Check PIP Version:
```bash
python3 -m pip -V
```

>Upgrade PIP if needed:
>```bash
>python3 -m pip install --upgrade pip
>```

Install OpenCDA dependencies:

```bash
cd ~/OpenCDA
conda create -n opencda python=3.10
conda activate opencda
pip install -r requirements.txt
```

Install SUMO:
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
```

Additional packages required for state estimation and DAROM submodule:

```bash
pip install "torch>=2.0.0" "torchvision>=0.15.0" "stable-baselines3>=2.3.0" \
    gymnasium "hydra-core>=1.3.0" "tensorboard>=2.13.0" \
    "scikit-learn>=1.3.0" ultralytics
```

### 4. Install the CARLA Python API

With the `opencda` conda environment still active:

```bash
export CARLA_HOME=~/carla   # adjust to your CARLA install path
export CARLA_VERSION=0.9.16
source ~/OpenCDA/setup.sh
```

- For CARLA built from source only:
  ```bash
  pip install $CARLA_HOME/Dist/CARLA_Shipping_4ca87e818/LinuxNoEditor/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-linux_x86_64.whl
  ```
### 5. Install OpenCDA

```bash
cd ~/OpenCDA
pip install -e .
```

---

## Map Installation

### Final Highway Model (FHWM)

FHWM is an earlier version of US 101 that is optimized for performance. The map is available for both Linux and Windows from the [Maps v1.0 Release Page](https://github.com/Juliang0729/Team-CARLA-Merging-Simulation-for-Autonomous-Vehicles_2026/releases/tag/maps).

- **Linux Installation:**

  1. Download `FHWM_Linux.tar.gz` from the release page.
  2. Copy the archive into your CARLA `Dist` directory:
     ```bash
     cp FHWM_Linux.tar.gz $CARLA_HOME/Dist/
     ```
  3. Run the CARLA asset importer:
     ```bash
     cd $CARLA_HOME
     bash ImportAssets.sh
     ```
     The script will detect and import the FHWM package automatically.

- **Windows Installation:**

  1. Download `FHWM_Windows.zip` from the release page.
  2. Extract the archive directly into your CARLA root directory at `$CARLA_HOME`.
  3. (**Do NOT overwrite existing files**) When prompted, select   "Skip" or "Keep existing" for any conflicts. Only new map-specific files should be added.
  
  After installation the `FHWM` map will appear in CARLA alongside the   default Town maps.

### US Highway 101 (US101)

The US101 map is compiled into a dedicated Linux-only CARLA build. It cannot be imported via `ImportAssets.sh` due to the map being embedded in the packaged engine content.

**Download from Hugging Face:** [US101_Map](https://huggingface.co/datasets/Jgross29/US101_Map)

```bash
wget https://huggingface.co/datasets/Jgross29/US101_Map/resolve/main/LinuxNoEditor.tar.xz
tar -xJf LinuxNoEditor.tar.xz -C ~/carla/Unreal/CarlaUE4/Saved/StagedBuilds/
```

Once extracted, the directory structure should match:

```
~/carla/Unreal/CarlaUE4/Saved/StagedBuilds/LinuxNoEditor/
├── CarlaUE4.sh
├── Manifest_NonUFSFiles_Linux.txt
├── Engine/
└── CarlaUE4/
    ├── Binaries/
    └── Content/
        ├── Carla/
        └── Paks/
            └── CarlaUE4-LinuxNoEditor.pak   ← contains US101 Map
```

The `run_us101_cosim.sh` launch script expects CARLA at this exact path. If you place the build elsewhere, update the `CARLA_DIR` variable at the top of the script.

---

## Running the Simulation

### US101 Co-Simulation (Recommended)

A convenient shell script handles starting CARLA and launching the OpenCDA co-simulation in one command:

```bash
cd ~/OpenCDA
bash run_us101_cosim.sh
```

This will:
1. Start the US101 CARLA packaged build in the background.
2. Wait ~15 seconds for CARLA to initialize.
3. Activate the `opencda` conda environment.
4. Run `python opencda.py -t single_us101_cosim --apply_ml`.

### FHWM Co-Simulation

Start a standard CARLA server first by running `./CarlaUE4.sh` at the distribution root, then:

```bash
conda activate opencda
cd ~/OpenCDA
python opencda.py -t single_fhwm_cosim --apply_ml
```

### Command-Line Options

| Flag | Description |
|:---:|:---:|
| `--apply_ml` | Enable ML/DL framework support (always required for DAROM) |
| `--state_estimator` | Enable 8-camera YOLO surround-view object detection and tracking |
| `--evaluation` | Enable performance metrics logging to `performance_metrics.csv` (automatically enables `--state_estimator`) |
| `-h`,`--help` | Returns complete list of command-line options with descriptions |

Example — run with full evaluation pipeline:

```bash
bash run_us101_cosim.sh --evaluation
```

---

## Optional Modules

### State Estimator

When `--state_estimator` is passed, `StateEstimator` attaches 8 RGB cameras in a surround-view ring around the ego vehicle. Each frame is processed by a YOLOv5m detector (`yolov5m.pt`, included in the repository root) to locate nearby vehicles. A lightweight LiDAR clustering pass at 4 Hz validates detections. Tracking activates only after the ego vehicle merges onto the mainline.

### Performance Evaluator

When `--evaluation` is passed, `PerformanceEvaluator` logs the following metrics to `performance_metrics.csv` every simulation step:

| Metric | Description |
|:---:|:---:|
| Simulator time offset | Offset between CARLA and SUMO simulation clocks |
| Control latency | Time to compute and apply DAROM actions |
| Lateral error | Ego deviation from target lane centerline |
| Detection count vs. SUMO count | Object detection recall proxy |
| Position error | YOLO detection error vs. SUMO ground truth (m) |
| Drift rate | Rate of position error change (m/s) |
| Collision events | Ego collision count |

---

## Project Structure

```
OpenCDA/
├── opencda.py                          # Main entry point
├── run_us101_cosim.sh                  # One-command US101 launch script
├── setup.sh                            # CARLA Python API installer
├── environment.yml                     # Conda environment definition
├── yolov5m.pt                          # YOLOv5m weights for state estimation
├── onRampMerging/                      # DAROM submodule (Amin Tabrizian)
│   ├── models/GRU-uniform-delay/       # Pretrained DAROM-GRU checkpoint
│   └── src/                            # RDMDP env, delay encoder, SAC agent
└── opencda/
    ├── scenario_testing/
    │   ├── single_us101_cosim.py       # US101 scenario runner
    │   ├── single_fhwm_cosim.py        # FHWM scenario runner
    │   ├── merging_controller.py       # DAROM agent wrapper
    │   ├── state_estimator.py          # 8-cam YOLO + LiDAR tracker
    │   ├── performance_evaluator.py    # Metrics logger
    │   ├── birdseye_camera.py          # Spectator bird's-eye view
    │   └── config_yaml/
    │       ├── single_us101_cosim.yaml # US101 scenario parameters
    │       └── single_fhwm_cosim.yaml  # FHWM scenario parameters
    └── assets/
        ├── US101_SUMO/                 # SUMO network + route files for US101
        └── FHWM_SUMO/                  # SUMO network + route files for FHWM
```

---

## Acknowledgements

- **DAROM / onRampMerging** — Amin Tabrizian, Zhitong Huang, Arsyi Aziz, Peng Wei (GWU): [Website](https://amin-tabrizian.github.io/DAROM/) · [Repo](https://github.com/amin-tabrizian/onRampMerging)
- **OpenCDA** — Runsheng Xu et al. (UCLA): [Repo](https://github.com/ucla-mobility/OpenCDA)
- **CARLA Simulator** — [carla.org](https://carla.org)
- **SUMO** — [Eclipse SUMO](https://sumo.dlr.de/)
- **NGSIM US Highway 101 Dataset** — [FHWA NGSIM](https://www.fhwa.dot.gov/publications/research/operations/07030/)
