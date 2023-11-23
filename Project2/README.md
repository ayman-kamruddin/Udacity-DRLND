# Project 2: Continuous Control

## Overview

This project is part of the Udacity Deep Reinforcement Learning Nanodegree and focuses on training a deep reinforcement learning agent to control a robotic arm to reach target locations in a Unity environment. The goal is to implement and train an agent using deep deterministic policy gradients (DDPG) to effectively control the robotic arm in a continuous action space.

## Project Structure

The project is organized into the following main components:

- **`main.py`**: Python containing the code for training the DDPG agent and exploring the environment.

- **`ddpg_agent.py`**: Python script defining the DDPG agent class.

- **`model.py`**: Python script defining the actor and critic network architectures used by the agent.

## Getting Started

Follow this step to get started with the Continuous Control project:

1. Run the `main.py` python file.

## Dependencies

Ensure you have at least the following dependencies installed:

- Python 3.6 or later
- PyTorch
- NumPy
- Unity ML-Agents (for the Reacher environment)

## Results

- **`model_actor.pth`**: Model weights for the actor network

- **`model_critic.pth`**: Model weights for the critic network

## Tech Specs

Vector Observation space type: continuous
Vector Observation space size (per agent): 33
Vector Action space type: continuous
Vector Action space size (per agent): 4
Number of agents: 20