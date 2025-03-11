#!/bin/bash
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# Run the Python script for training
# echo "Running Python script for training ..."
# python3 train.py -run-name lr_exploration -algo TD3 -lr 0.03

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo TD3 -lr 0.003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo TD3 -lr 0.0003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo TD3 -lr 0.00003


# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo PPO -lr 0.003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo PPO -lr 0.0003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration -algo PPO -lr 0.00003


#PPO-LSTM
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_003 -algo PPO_LSTM -lr 0.003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_0003 -algo PPO_LSTM -lr 0.0003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_00003 -algo PPO_LSTM -lr 0.00003


#TD3
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_003_dt_0_05 -algo TD3 -lr 0.0003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "Learning starts increased to 20000 on TD3 "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_003_dt_0_05 -algo TD3 -lr 0.003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "Learning starts increased to 20000 on TD3 "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_00003_dt_0_05 -algo TD3 -lr 0.00003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "Learning starts increased to 20000 on TD3 "

# #PPO
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_003 -algo PPO -lr 0.003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "PPO batch size increased (was fixed to 256 before)"

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_0003 -algo PPO -lr 0.0003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "PPO batch size increased (was fixed to 256 before)"

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_00003 -algo PPO -lr 0.00003 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "PPO batch size increased (was fixed to 256 before)"

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_03 -algo PPO -lr 0.03 -env-notes "delta t changed from 0.001 to 0.05 in ros config. " -algo-notes "PPO batch size increased (was fixed to 256 before)"



#PPO-LSTM

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_003 -algo PPO_LSTM -lr 0.003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_0003 -algo PPO_LSTM -lr 0.0003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_00003 -algo PPO_LSTM -lr 0.00003

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lr_exploration_0_03 -algo PPO_LSTM -lr 0.03


#TD3 - lidar noise - 09/08/24 completed!!!!
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo TD3 -lr 0.0003 --lidar-noise -lidar-noise-size 3 -env-notes "Lidar noise size 3x3 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo TD3 -lr 0.0003 --lidar-noise -lidar-noise-size 5 -env-notes "Lidar noise size 5x5 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo TD3 -lr 0.0003 --lidar-noise -lidar-noise-size 7 -env-notes "Lidar noise size 7x7 " -algo-notes " "




# #PPO - lidar noise - 09/08/24 set off
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name PPO_Clean_500k -algo PPO -lr 0.003 -max-training-steps 500000 -env-notes "clean PPO 500k " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name TD3_Clean_500k -algo TD3 -lr 0.0003 -max-training-steps 500000 -env-notes "clean TD3 500k " -algo-notes " "

# #PPO lidar noise!
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo PPO -lr 0.003 --lidar-noise -lidar-noise-size 3 -max-training-steps 100000 -env-notes "Lidar noise size 3x3 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo PPO -lr 0.003 --lidar-noise -lidar-noise-size 5 -max-training-steps 100000 -env-notes "Lidar noise size 5x5 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name lidar_noise -algo PPO -lr 0.003 --lidar-noise -lidar-noise-size 7 -max-training-steps 100000 -env-notes "Lidar noise size 7x7 " -algo-notes " "


#camera noise - ppo (were rawdogging the LR FYI for all camera obs spaces)
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.003 --camera-noise -camera-noise-size 3 -max-training-steps 100000 -env-notes "camera noise size 3x3 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.003 --camera-noise -camera-noise-size 5 -max-training-steps 100000 -env-notes "camera noise size 5x5 " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.003 --camera-noise -camera-noise-size 7 -max-training-steps 100000 -env-notes "camera noise size 7x7 " -algo-notes " "


#camera noise - td3
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.0003 --camera-noise -camera-noise-size 3 -max-training-steps 100000 -env-notes "camera noise size 3x3 " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.0003 --camera-noise -camera-noise-size 5 -max-training-steps 100000 -env-notes "camera noise size 5x5 " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.0003 --camera-noise -camera-noise-size 7 -max-training-steps 100000 -env-notes "camera noise size 7x7 " -algo-notes " "



#Extra - LR exploration for camera (PPO)
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.03 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.0003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 100000 -env-notes "Camera obs LR exploration" -algo-notes " "


#Extra - LR exploration for camera (TD3)
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.03 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.0003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo TD3 -lr 0.00003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "


#Extra - LR exploration for camera (PPO_LSTM)
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO_LSTM -lr 0.03 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO_LSTM -lr 0.003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO_LSTM -lr 0.0003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 100000 -env-notes "Camera obs LR exploration " -algo-notes " "


#Extra - LR exploration for lidar (PPO_LSTM)
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "lidar" -algo PPO_LSTM -lr 0.03 -max-training-steps 100000 -env-notes "Lidar obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "lidar" -algo PPO_LSTM -lr 0.003 -max-training-steps 100000 -env-notes "Lidar obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "lidar" -algo PPO_LSTM -lr 0.0003 -max-training-steps 100000 -env-notes "Lidar obs LR exploration " -algo-notes " "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name camera_noise -obs "lidar" -algo PPO_LSTM -lr 0.00003 -max-training-steps 100000 -env-notes "Lidar obs LR exploration " -algo-notes " "





#NB: I had to lower buffer size for TD3 camera to 10,000!But this does not affect the lidar one