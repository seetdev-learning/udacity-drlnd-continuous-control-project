# Udacity Deep Reinforcement Learning Nanodegree Navigation Project
This repository contains the code that is submitted as project 2 (p2 - Continuous Control) of Udacity's Deep Reinforcement Learning Nanodegree

## Project Overview
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training
For this project, we will provide you with two separate versions of the Unity environment:

The first version contains a single agent.
The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

### Solving the Environment
Note that your project submission need only solve one of the two versions of the environment.

#### Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.

### Folder Structure

## Introduction

## Pre-requisites

This project runs as a Jupyter notebook that is ran in a mini-conda enviroment. The following describes the instructions to run the notebook.

### Installing Unity Hub
Follow the instructions and install the Unity Hub that fits your operating system from https://store.unity.com/download

### Installing Miniconda or Anaconda

Anaconda is a free and open-source distribution of Python and R programming languages for scientific computing, that aims to simplify package management and deployment. Miniconda is a slim-down version of Anaconda with less packages included within the installer. The 2 should work similarly with just a different installer download. And Miniconda might require more manual installation of packages. The instructions for installation is found in https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation and it supports Windows, MacOS and Linux.

### Constructing the Conda environment to run the project

This project specific packages for execution and the versions are locked down to prevent version compatibility issues. A conda environment is constructed to host it so that the host machine is not impacted.

#### Create the Conda environment

Run `conda create --name drlnd-p2 python=3.6` (This is only ran once on every host)

#### Activate the Conda environment

Run `conda activate --name drlnd-p2` (This is ran everytime before activating the Jupyter Notebook server)

#### Install the require Python packages

1. Run `conda install -c conda-forge -c pytorch Pillow matplotlib numpy jupyter pytest docopt pyyaml pytorch pandas scipy ipykernel` to install the following packages:

    - tensorflow
    - protobuf
    - grpcio
    - Pillow
    - matplotlib 
    - numpy
    - jupyter 
    - pytest
    - docopt 
    - pyyaml 
    - pytorch
    - pandas 
    - scipy 
    - ipykernel
2. Run `pip install unityagents` to install Unity Agents

## Folder

```
.
├── Continuous_Control.ipynb    # Original notebook
├── LICENSE
├── README.md                   # This file
├── Report.pdf                  # pdf export of the runner's last run
├── Report.ipynb                # Notebook with runner
├── checkpoint_actor.pth        # Save weights file of Actor
├── checkpoint_critic.pth       # Save weights file of Critic
├── ddpg                        # ddpg agent folder
│   ├── __init__.py             # init file to make ddpg a package
│   ├── agent.py                # agent implementaion
│   └── model.py                # model file for Actor and Critic
├── unity_environments          # repo contains various environments for single and multiple agents on different OS
│   ├── multi_agent             # check the subfolder for all environment files used for multi agent algorithms
│   └── single_agent            # check the subfolder for all environment files used for single agent algorithms
├── python                      # files related to dependencies installation on Udacity workspace
│   ├── ....
└── replaybuffer.py             # code file for memory buffer
```
## Environment File Download

If you wish to download the unity environment files please follow the instructions below:

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

## Running the agent

1. Setup and activate the conda environment setup 
2. Open Report.ipynb and set the `workspace` and `env_file_name` according to the runtime
3. Run all cells in the notebook
