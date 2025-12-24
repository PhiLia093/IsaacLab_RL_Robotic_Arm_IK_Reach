# Isaac Lab Reinforcement Learning: Robotic Arm Reach
<img width="2686" height="1459" alt="image_2025-12-23_17-38-00" src="https://github.com/user-attachments/assets/fdbe6524-c103-490f-975a-f8522655a361" />



## Overview
This project implements a Reinforcement Learning (RL) environment using NVIDIA Isaac Lab (based on Isaac Sim) to train a 6-DOF Universal Robots arm. The goal is to train a policy using PPO that allows the robot end-effector to track a randomly moving target in 3D space.

The project focuses on engineering a robust Markov Decision Process (MDP), designing shaped reward functions for high-precision control, and utilizing parallel simulation for efficient training.

## Tech Stack
- **Framework:** NVIDIA Isaac Lab, Omniverse
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Language:** Python, PyTorch
- **Physics:** PhysX

## Key Engineering Features

### 1. MDP & Environment Design
- **Modular Architecture:** Utilized Isaac Lab's configuration class system to decouple scene, action, and observation definitions for reusability.
- **Action Space:** Configured implicit actuators using a PD control scheme with relative target offsets (Delta Control).
- **Observation Space:** Assembled a multi-modal observation vector including joint positions, velocities, and target error. Integrated Gaussian noise injection to simulate sensor uncertainty and improve Sim-to-Real robustness.

### 2. Custom Reward Engineering
Designed a composite reward function to optimize for both tracking speed and motion quality:
- **Coarse Tracking:** Implemented a negative L2-norm distance penalty to guide the robot toward the target area.
- **Fine-Grained Precision:** Developed a custom Tanh kernel reward function to amplify gradients when the end-effector is close to the target (Sweet Spot), effectively minimizing steady-state error.
- **Regularization:** Penalized high joint velocities and action rates (jerk) to prevent mechanical oscillation and ensure hardware-safe trajectories.

### 3. Curriculum Learning
Implemented a dynamic training schedule that linearly increases penalty weights for motion smoothness over time. This allows the agent to explore the state space aggressively in early stages while converging on smooth, stable control policies by the end of training.

### 4. Massively Parallel Simulation
Configured the environment to run 2048+ parallel instances on a single GPU using Headless mode. This significantly accelerates data collection, compressing hours of training time into minutes.

## Project Structure
- `reach_env_cfg.py`: Main environment configuration aggregating scene layout, MDP settings, and simulation parameters (dt, decimation).
- `ur_gripper.py`: Robot asset configuration defining initial joint states and actuator stiffness/damping (PD gains).
- `rewards.py`: Custom PyTorch implementations of the L2 and Tanh error calculation logic.
- `actions.py`: Definitions for the joint position control interface mapping neural net outputs to motor targets.

## Usage
To train the policy:
```bash
python train.py --task=Isaac-Reach-UR-v0 --num_envs=2048 --headless
