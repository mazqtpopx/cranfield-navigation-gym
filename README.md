# Cranfield Navigation Gym
## A ROS-based Gymnasium for Training DRL Agents in Adversiarially Perturbed Navigation Scenarios
This repository provides a **ROS-Gymnasium Wrapper** designed for developing and training Deep Reinforcement Learning (DRL) models using the Robot Operating System (ROS) for adversially perturbed (i.e. sensor denial and noise sensor areas in the environment) navigation scenarios. The package integrates with ROS (Robot Operating System) for real-time communication and Gazebo for 3D simulation, and it supports popular DRL algorithms such as TD3 and PPO through [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). 

This repo contains the code for the paper **Benchmarking Deep Reinforcement Learning for
Navigation in Denied Sensor Environments**. Please refer to our [preprint](https://arxiv.org/abs/2410.14616) for the results and conducted experiments. This environment map and the navigation screnario builds on top of the [DRL robot navigation](https://github.com/reiniscimurs/DRL-robot-navigation) repository and the publication [Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9645287?source=authoralert).

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Run Publication experiments](#run-publication-experiments)
  - [Basic Usage](#basic-usage)
    - [Configurations](#configurations)
  - [Package API](#package-api)
- [Future Developments](#future_developments)
- [License](#license)



## Overview
This project allows integration between Gymnasium and ROS, enabling reinforcement learning models to be trained on navigation tasks in adversarial environments. It simplifies the process of managing Gazebo simulations and handling communication between the simulation and the RL agents. 

The environment includes support for various sensor configurations (e.g., Lidar, Camera) and incorporates sensor perturbations to simulate real-world challenges.

DRL algorithms supported:
  - **TD3**: Twin Delayed Deep Deterministic Policy Gradients
  - **PPO**: Proximal Policy Optimization
  - **PPO-LSTM**: Recurrent variant of PPO
  - **DreamerV3**: This was used to generate results in the paper but an interface is not yet published

## Features

This project offers a customizable *ROS-Based Gym wrapper* for training deep reinforcement learning (DRL) models in simulated robot environments. The core features are driven by two primary scripts:

### 1. **ROS-Gymnasium Wrapper**
A custom Gymnasium environment that extends Gym’s functionality allowing for the gymnasium classes to interface with ROS and Gazebo. 

Key functionality is provided by:
- **`ros_interface.py`**: This class exposes the ROS interface, allowing real-time communication with ROS, handling topics like sensor data (Lidar, Camera) and robot control (velocity commands). It also supports launching, resetting, and closing ROS nodes, as well as managing the simulated models (i.e robot and goal target), or other functionalities like updating adversarial attack areas (sensor denial/sensor noise areas) during simulation.

- **`drl_robot_navigation.py`**: This is the main script that defines a custom Gym environment for robot navigation, integrating action and observation spaces (Lidar, Camera), and handles reward structures and episode termination conditions. The main functionalities from gymnasium are applied here (i.e. step, reset). Additionally functions for provided rewards, termination of episode etc. are performed here.

### 2. **Adversarial Sensor Attacks Simulation**
The wrapper allows for customizable noise to be applied to Lidar and Camera sensors, simulating sensor perturbations such as sensor failures or attacks. The type and level of noise can be configured via YAML config files (see [Configuration](#configuration)). 

### 3. **Configurable Environment**
The system is configurable through YAML files, enabling switching between different algorithms (TD3, PPO, PPO-LSTM), sensor setups, and sensor perturbations. 

### 5. **Support for Multiple DRL Algorithms**
The wrapper integrates with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3), supporting DRL algorithms like TD3, PPO, and PPO-LSTM. 

## Installation

### Prerequisites
This repo was developed and tested using Ubuntu 20.04 and Python 3.8.10. We cannot guarantee that it will work on other configurations.  
- **[ROS Noetic](http://wiki.ros.org/noetic/Installation)**: Ensure ROS is installed (including Gazebo 11).
- **Python 3.8+**: This was tested with python 3.8.10. NB: For 3.9+ you will have to ```pip3 install netifaces-plus```
- **[PyTorch](https://pytorch.org/get-started/locally/)**
- **[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)**
- **[ROS catkin workspace](https://github.com/parisChatz/ros-rl-env)**: This wrapper needs a ros package to handle the bringup of a robot, the controllers, and everything necessary to spawn a robot, and the ROS/Gazebo simulation environment. For this work we provide the ros workspace that uses the a mobile robot (Pioneer 3-DX) with the necessary packages for bringup, control, and simulation of this robot.
      
    ```bash
    git clone https://github.com/parisChatz/ros-rl-env.git
    cd ~/ros-rl-env/catkin_ws/src
    rosdep install --from-paths src --ignore-src -r -y
    catkin_make_isolated
    ```

### Installing the Package
1. Clone this repository and install it as a Python package:
    ```bash
    git clone https://github.com/mazqtpopx/cranfield-navigation-gym.git
    cd ~/cranfield-navigation-gym
    python3 -m pip install .
    ```

2. Install additional dependencies:
    ```bash
    python3 -m pip install -r ~/cranfield-navigation-gym/requirements.txt
    ```

## Usage

### Run publication experiments
To reproduce the publications results you only need navigate to the directory and run the appropriate bash experiment file:

```
cd ~/cranfield-navigation-gym/cranavgym/tests/training_scripts/
```  

#### Train Camera Denial (PPO)
Run the file ```train_camera_denial.sh```:
```
./train_camera_denial.sh
```
#### Train, LR Exploration
Run the file ```train_lr_exploration.sh```:
```
./train_lr_exploration.sh
```
#### Evaluate PPO Camera Denial (NB: will need to update your paths based on trained models!)
Run the file ```evaluate_PPO_camera_denial.sh```:
```
./evaluate_PPO_camera_denial.sh
```

When the experiments are done you will see the folder ```~/cranfield-navigation-gym/cranavgym/log_dir``` where you can find logs for the experiments.

### Basic Usage
This wrapper allows you to easily train DRL algorithms in a ROS. Assuming that there is a ros workspace (in our case *ros-rl-env*), a roslaunch file (```~/cranfield-navigation-gym/cranavgym/ros_interface/DRLNav.launch```) should be provided that launches all the necessary packages for the robot simulation that the ros interface (```~/cranfield-navigation-gym/cranavgym/ros_interface/ros_interface.py```) will make use of. This inteface executes the ros launch file, creates appropriate publishers and subscribers, and handles the ros communication and gazebo simulation.

This interface is used by the wrapper (```~/cranfield-navigation-gym/cranavgym/envs/drl_robot_navigation.py```) to handle the simulation according to the training of the DRL agents.

#### Configurations
There are 3 configuration files in the ```~/cranfield-navigation-gym/cranavgym/configs``` folder that are used by the wrapper and the interface. 

- **env_config.yaml**: Includes general configurations for scenario settings (i.e. max episode step, adversary configurations).
- **rl_config.yaml**: Includes configurations about the DRL algorithms, as well as the paths for logging results.
- **ros_interface_config.yaml**: Includes configurations for ros and gazebo.

## Custom training
cd to the tests dir:
```
cd ~/cranfield-navigation-gym/cranavgym/tests/
```
Source the setup_env.sh (this sets up the environment - you can also run the commands inside manually):
```
cd ~/cranfield-navigation-gym/cranavgym/tests/
```
Train the model
```bash
    python3 train.py
```
The script loads the configurations from the 3 config files and runs the training on Gazebo.

When executing ```train.py``` you can also put arguments that will overwrite the arguments of the config files.
Do the following to look into which configurations can be added as arguments when running train.py:
```bash
    python3 train.py --help
```

### Add adversaries in the environment
First decide which sensors should be attacked. Currently there are only adversarial pertubations in the lidar and camera sensors.
Inside the env_config.yaml, under the drl_robot_navigation, you can find the configurations for the noises. Change them accordingly.

Introduce adversaries in the camera sensors during training you can either change the env_config.yaml and run ```python3 train.py``` or without changing the configs you can run:
```bash
python3 train.py --camera-noise -camera-noise-size 3
```

### Choose different Map
#### No obstacle Map
To set the map to the evaluation no obstacle map then navigate to ```~/cranfield-navigation-gym/cranavgym/ros_interface/DRLNav.launch```, and change the line in the file:
```
  <arg name="world_name" value="$(find multi_robot_scenario)/worlds/TD3.world"/>
``` 
to
```
  <arg name="world_name" value="$(find multi_robot_scenario)/worlds/training_big_rect.world"/>
``` 

#### Custom ROS Map
Navigate to ```~/cranfield-navigation-gym/cranavgym/ros_interface/DRLNav.launch```, and change the line in the file to point at another .world file:
```
  <arg name="world_name" value="$(find multi_robot_scenario)/worlds/TD3.world"/>
``` 
Make sure to also change the ```min_xy``` and ```max_xy``` configurations in the```~/cranfield-navigation-gym/cranavgym/configs/ros_interface_config.yaml``` to accomodate to the possible change in map size. This size is used by the interface to spawn and move the models of the Gazebo simulation.

## Future Developments

This project is actively being developed, and several key features are planned for future releases:

1. **Moving Goal Support**: Adding support for dynamic goals that move in the environment during episodes, creating more complex navigation challenges for reinforcement learning agents.
2. **Advanced Collision Handling**: Enhancing collision detection and include soft-body obstacles that the robot can push aside, increasing the diversity of possible environments.
3. **Dynamic Obstacles**: Introducing moving obstacles to simulate more realistic environments where the robot must navigate around objects in motion.
4. **Support for Additional Sensors**: Expanding sensor simulation to include more types of sensors (e.g., ultrasonic, depth cameras) for a richer environment setup.
5. **Multi-Agent Support**: Implementing functionality for multi-agent environments, enabling multiple robots to collaborate or compete within the same environment.
6. **Multiple Adversarial Attack areas**: Add support for multiple adversarial attack areas of the same type in the environment.
7. **Dynamic ROS topics definition**: The ROS topics that are used to for observations (e.g. odometry, velodyne, camera) and for the actions (i.e. cmd_vel), are hardcoded in the ```ros_interface.py```.
8. **ROS2 Support**

Contributions and suggestions for new features are always welcome. If you have ideas or feedback, feel free to open an issue or submit a pull request!


#### Troubleshooting
There are two main parts: ROS/Gazebo (Env) and Python/Pytorch (Agent)
First, cd to the tests dir:
```
cd ~/cranfield-navigation-gym/cranavgym/tests/
```  
Run test_envs.sh
```
./test_envs.sh
```
If test_envs runs correctly (i.e. gazebo and rviz are shown and the robot takes random actions), this suggests an error with the stable baselines/pytorch.
If test_envs does not run correctly, it suggests an error with your ROS/Gazebo installation.


## Citations
If you use this repo for academic work please consider citing our [preprint](https://arxiv.org/abs/2410.14616):
```
@misc{wisniewski2024benchmarkingdeepreinforcementlearning,
      title={Benchmarking Deep Reinforcement Learning for Navigation in Denied Sensor Environments}, 
      author={Mariusz Wisniewski and Paraskevas Chatzithanos and Weisi Guo and Antonios Tsourdos},
      year={2024},
      eprint={2410.14616},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.14616}, 
}
```

Also as this repo was built on top of Cimurs et al. work ([DRL robot navigation](https://github.com/reiniscimurs/DRL-robot-navigation)), please cite them as well: 
```
@ARTICLE{9645287,
  author={Cimurs, Reinis and Suh, Il Hong and Lee, Jin Han},
  journal={IEEE Robotics and Automation Letters}, 
  title={Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning}, 
  year={2022},
  volume={7},
  number={2},
  pages={730-737},
  doi={10.1109/LRA.2021.3133591}}
```

## License

This project is licensed under the **MIT License**.
