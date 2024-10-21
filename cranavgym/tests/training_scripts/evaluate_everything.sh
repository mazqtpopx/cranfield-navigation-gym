#!/bin/bash
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh

#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3_default_map -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5_default_map -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7_default_map -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_0x0 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_0x0 -obs camera -algo PPO -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_0x0 -obs camera -algo PPO -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "








PPO_lstm trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_baseline_default_map -obs camera -algo PPO_LSTM -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_3x3_default_map -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_5x5_default_map -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_120510_camera_static_goal_noise_0/model/best_model.zip -run-name default_7x7_default_map -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#PPO_lstm 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_172631_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_lstm_3x3_default_map_0x0 -obs camera -algo PPO_LSTM -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_172631_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_lstm_3x3_default_map_3x3 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_172631_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_lstm_3x3_default_map_5x5 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_172631_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_lstm_3x3_default_map_7x7 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


#PPO_lstm 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_224917_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_lstm_5x5_default_map_0x0 -obs camera -algo PPO_LSTM -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_224917_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_lstm_5x5_default_map_3x3 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_224917_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_lstm_5x5_default_map_5x5 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240828_224917_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_lstm_5x5_default_map_7x7 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#PPO_lstm 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240829_040421_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_lstm_7x7_default_map_0x0 -obs camera -algo PPO_LSTM -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240829_040421_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_lstm_7x7_default_map_3x3 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240829_040421_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_lstm_7x7_default_map_5x5 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_LSTM_20240829_040421_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_lstm_7x7_default_map_7x7 -obs camera -algo PPO_LSTM -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "




#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_baseline_default_map -obs camera -algo TD3 -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_3x3_default_map -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_5x5_default_map -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240904_062038_camera_static_goal_noise_500k/model/best_model.zip -run-name default_7x7_default_map -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_130619_camera_static_goal_noise_3x3/model/best_model.zip -run-name td3_3x3_default_map_0x0 -obs camera -algo TD3 -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_130619_camera_static_goal_noise_3x3/model/best_model.zip -run-name td3_3x3_default_map_3x3 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_130619_camera_static_goal_noise_3x3/model/best_model.zip -run-name td3_3x3_default_map_5x5 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_130619_camera_static_goal_noise_3x3/model/best_model.zip -run-name td3_3x3_default_map_7x7 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_190639_camera_static_goal_noise_5x5/model/best_model.zip -run-name td3_5x5_default_map_0x0 -obs camera -algo TD3 -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_190639_camera_static_goal_noise_5x5/model/best_model.zip -run-name td3_5x5_default_map_3x3 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_190639_camera_static_goal_noise_5x5/model/best_model.zip -run-name td3_5x5_default_map_5x5 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240906_190639_camera_static_goal_noise_5x5/model/best_model.zip -run-name td3_5x5_default_map_7x7 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240907_003004_camera_static_goal_noise_7x7/model/best_model.zip -run-name td3_7x7_default_map_0x0 -obs camera -algo TD3 -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240907_003004_camera_static_goal_noise_7x7/model/best_model.zip -run-name td3_7x7_default_map_3x3 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240907_003004_camera_static_goal_noise_7x7/model/best_model.zip -run-name td3_7x7_default_map_5x5 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240907_003004_camera_static_goal_noise_7x7/model/best_model.zip -run-name td3_7x7_default_map_7x7 -obs camera -algo TD3 -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "





#LIDAR

#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_baseline_default_map -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_3x3_default_map -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_5x5_default_map -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_7x7_default_map -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_0x0 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_3x3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_5x5 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_7x7 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_0x0 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_3x3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_5x5 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_7x7 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_0x0 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_3x3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_5x5 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_7x7 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_baseline_default_map -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_3x3_default_map -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_5x5_default_map -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_7x7_default_map -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_0x0 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_3x3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_5x5 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_7x7 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_0x0 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_3x3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_5x5 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_7x7 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_0x0 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_3x3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_5x5 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_7x7 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "











#LIDAR

#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep1 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_0x0_rep1 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_3x3_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_5x5_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_7x7_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_0x0_rep1 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_3x3_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_5x5_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_7x7_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_0x0_rep1 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_3x3_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_5x5_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_7x7_rep1 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep2 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_0x0_rep2 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_3x3_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_5x5_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_7x7_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_0x0_rep2 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_3x3_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_5x5_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_7x7_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "



#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_0x0_rep2 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_3x3_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_5x5_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_7x7_rep2 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "



#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep3 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_0x0_rep3 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_3x3_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_5x5_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_7x7_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_0x0_rep3 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_3x3_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_5x5_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_7x7_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_0x0_rep3 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_3x3_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_5x5_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_7x7_rep3 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#TD3 trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep4 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240810_161616_TD3_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#TD3 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_0x0_rep4 -obs lidar -algo TD3 -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_3x3_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_5x5_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_164017_lidar_noise/model/best_model.zip -run-name td3_3x3_default_map_7x7_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#TD3 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_0x0_rep4 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_3x3_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_5x5_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240808_221049_lidar_noise/model/best_model.zip -run-name td3_5x5_default_map_7x7_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#TD3 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_0x0_rep4 -obs lidar -algo TD3 -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_3x3_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_5x5_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/TD3_20240809_045009_lidar_noise/model/best_model.zip -run-name td3_7x7_default_map_7x7_rep4 -obs lidar -algo TD3 -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep1 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_0x0_rep1 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_3x3_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_5x5_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_7x7_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_0x0_rep1 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_3x3_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_5x5_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_7x7_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_0x0_rep1 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_3x3_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_5x5_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_7x7_rep1 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "





#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep2 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_0x0_rep2 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_3x3_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_5x5_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_7x7_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_0x0_rep2 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_3x3_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_5x5_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_7x7_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_0x0_rep2 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_3x3_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_5x5_rep2 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_7x7_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep3 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_0x0_rep3 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_3x3_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_5x5_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_7x7_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_0x0_rep3 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_3x3_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_5x5_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_7x7_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_0x0_rep3 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_3x3_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_5x5_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_7x7_rep3 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "




#PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_baseline_default_map_rep4 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_3x3_default_map_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_5x5_default_map_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240809_163032_PPO_Clean_500k/model/best_model.zip -run-name default_7x7_default_map_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes " "

#PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_0x0_rep4 -obs lidar -algo PPO -lr 0.0003  -env-notes "lidar obs, evaluation " -algo-notes " "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_3x3_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_5x5_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_163134_lidar_noise/model/best_model.zip -run-name ppo_3x3_default_map_7x7_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_0x0_rep4 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_3x3_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_5x5_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240811_211526_lidar_noise/model/best_model.zip -run-name ppo_5x5_default_map_7x7_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_0x0_rep4 -obs lidar -algo PPO -lr 0.0003  -lidar-noise-size 0 -env-notes "lidar obs, evaluation " -algo-notes " cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_3x3_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 3  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_5x5_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 5  -env-notes "lidar obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240812_021331_lidar_noise/model/best_model.zip -run-name ppo_7x7_default_map_7x7_rep4 -obs lidar -algo PPO -lr 0.00003 --lidar-noise -lidar-noise-size 7  -env-notes "lidar obs, evaluation " -algo-notes "cam "



#Repeat all PPO runs 4 times 
# #PPO trained on 0x0 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map_1 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map_2 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map_3 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map_4 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3_default_map_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3_default_map_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3_default_map_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_3x3_default_map_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5_default_map_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5_default_map_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5_default_map_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_5x5_default_map_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7_default_map_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7_default_map_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7_default_map_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_7x7_default_map_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


# #PPO 3x3 
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_0x0_1 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_0x0_2 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_0x0_3 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_0x0_4 -obs camera -algo PPO -lr 0.0003 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_3x3_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_3x3_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_3x3_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_3x3_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_5x5_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_5x5_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_5x5_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_5x5_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_7x7_1 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_7x7_2 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_7x7_3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_175042_camera_static_goal_noise_3x3/model/best_model.zip -run-name ppo_3x3_default_map_7x7_4 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


#PPO 5x5
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_0x0 -obs camera -algo PPO -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240827_223006_camera_static_goal_noise_5x5/model/best_model.zip -run-name ppo_5x5_default_map_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

#PPO 7x7
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_0x0 -obs camera -algo PPO -lr 0.0003 --static-goal -camera-noise-size 0 -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_3x3 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 3 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_5x5 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 5 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240828_034223_camera_static_goal_noise_7x7/model/best_model.zip -run-name ppo_7x7_default_map_7x7 -obs camera -algo PPO -lr 0.00003 --camera-noise -camera-noise-size 7 --static-goal -env-notes "Camera obs, evaluation " -algo-notes "cam "


