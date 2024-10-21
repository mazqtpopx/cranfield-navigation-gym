#!/bin/bash
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
# python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_default_baseline -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_baseline -obs camera -algo PPO_LSTM -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
# python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_default_baseline -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_3x3 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_5x5 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_7x7 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "




source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_baseline -obs camera -algo TD3 -lr 0.00003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_3x3 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_5x5 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "

source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path /home/leo/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_7x7 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "TD3 cam "


