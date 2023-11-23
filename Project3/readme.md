# Project 3: Collaboration and Competition

## Overview

This project is part of the Udacity Deep Reinforcement Learning Nanodegree and focuses on training two agents to play tennis collaboratively in a Unity environment. The goal is to implement and train multi-agent reinforcement learning algorithms to enable the agents to learn effective strategies for playing tennis together.

## Project Structure

The project is organized into the following main components:

- **`main.py`**: - **`main.py`**: Python containing the code for training the multi-agent DDPG agent and exploring the environment.

- **`ddpg_agent.py`**: Python script defining the DDPG agent class for multi-agent scenarios.

- **`model.py`**: Python script defining the actor and critic network architectures used by the multi-agent DDPG agent.


## Getting Started

Follow this step to get started with the Collaboration and Competition project:

1. Run the `main.py` python file.


## Dependencies

Ensure you have the following dependencies installed:

- Python 3.6 or later
- PyTorch
- NumPy
- Unity ML-Agents (for the Tennis environment)


## Results

- **`model_actor.pth`**: Model weights for the actor network

- **`model_critic.pth`**: Model weights for the critic network

## Tech Specs

Vector Observation space type: continuous

Vector Observation space size (per agent): 8

Number of stacked Vector Observation: 3

Vector Action space type: continuous

Vector Action space size (per agent): 2

Number of agents: 2
