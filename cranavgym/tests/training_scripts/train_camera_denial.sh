#!/bin/bash
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name static_goal_camera -obs "camera" -algo PPO -lr 0.0003 --static-goal -max-training-steps 150000 -env-notes "Static goal, camera observation" -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name static_goal_camera_small_map -obs "camera" -algo PPO -lr 0.0003 --static-goal -map-bounds 2.5 -max-training-steps 150000 -env-notes "Static goal, 2.5 map bounds" -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name ppo_lstm_lr_exploration_0_0003 -obs "camera" -algo PPO_LSTM -lr 0.0003 -max-training-steps 150000 -env-notes "Camera obs LR exploration. Delta t 0.01, max steps 250 " -algo-notes "Updated n_steps and batch size to 512"

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_0 -obs "camera" -algo TD3 -lr 0.0003 --static-goal -max-training-steps 100000 -env-notes "Delta t 0.05, max steps 50 " -algo-notes ""

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_3x3 -obs "camera" -algo PPO -lr 0.0003 --camera-noise -camera-noise-size 3 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_5x5 -obs "camera" -algo PPO -lr 0.0003 --camera-noise -camera-noise-size 5 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_7x7 -obs "camera" -algo PPO -lr 0.0003 --camera-noise -camera-noise-size 7 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_0 -obs "camera" -algo PPO_LSTM -lr 0.0003 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_3x3 -obs "camera" -algo PPO_LSTM -lr 0.0003 --camera-noise -camera-noise-size 3 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_5x5 -obs "camera" -algo PPO_LSTM -lr 0.0003 --camera-noise -camera-noise-size 5 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_static_goal_noise_7x7 -obs "camera" -algo PPO_LSTM -lr 0.0003 --camera-noise -camera-noise-size 7 --static-goal -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.05, max steps 50" -algo-notes "Updated n_steps and batch size to 256"

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 sb3_test.py -run-name camera_500k -obs "camera" -algo PPO -lr 0.0003 -max-training-steps 500000 -env-notes "Camera obs LR exploration. Delta t 0.01, max steps 250" -algo-notes "Updated n_steps and batch size to 512"



#Extra - LR exploration for camera (PPO_LSTM)



# source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 sb3_test.py -run-name ppo_lstm_lr_exploration_0_00003 -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.01, max steps 250 " -algo-notes "Updated n_steps and batch size to 512"

# source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 sb3_test.py -run-name ppo_lstm_lr_exploration_0_03 -obs "camera" -algo PPO_LSTM -lr 0.03 -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.01, max steps 250 " -algo-notes "Updated n_steps and batch size to 512"

# source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3 sb3_test.py -run-name ppo_lstm_lr_exploration_0_003 -obs "camera" -algo PPO_LSTM -lr 0.003 -max-training-steps 100000 -env-notes "Camera obs LR exploration. Delta t 0.01, max steps 250 " -algo-notes "Updated n_steps and batch size to 512"
