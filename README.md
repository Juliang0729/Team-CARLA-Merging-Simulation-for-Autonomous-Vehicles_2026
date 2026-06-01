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
  - [Final Highway World Model (FHWM)](#final-highway-world-model-fhwm)
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

## Architecture
