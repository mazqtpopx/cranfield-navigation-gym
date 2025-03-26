#!/bin/bash
source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh




#25/03
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 5000000 --no-static-goal --no-frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "


source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3.10 train.py -run-name PPO_random_goal_continuous_act_noise -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 8000000 --no-static-goal --no-frame-stack --camera-noise -camera-noise-size 3 -env-notes "Camera obs LR exploration " -algo-notes " "



source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3.10 train.py -run-name PPO_random_goal_continuous_act_noise_retrain -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 8000000 --no-static-goal --no-frame-stack --camera-noise -camera-noise-size 3 -env-notes "Camera obs LR exploration " -algo-notes " " -load-model-path "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250326_153225_PPO_random_goal_continuous_act_lr/model/best_model.zip"


#Cone only obstacles + 
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_cone_obstacles_noise_retrain -obs "camera" -algo PPO --camera-noise -lr 0.000003 -max-training-steps 40000000 --no-frame-stack -env-notes "Camera obs LR exploration " --no-static-goal -algo-notes " " -load-model-path "/home/leo/cranfield-navigation-gym/log_dir/PPO_20250318_183507_Flight_arena_model_no_fs/model/best_model.zip"



# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_model_no_fs -obs "camera" -algo PPO_LSTM -lr 0.000003 -max-training-steps 10000000 --no-frame-stack --static-goal -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_model_no_fs_retrain_rand_goal_2_boxes -obs "camera" -algo PPO_LSTM -lr 0.000003 -max-training-steps 10000000 --no-frame-stack -env-notes "Camera obs LR exploration " --no-static-goal -algo-notes " " -load-model-path "/home/leo/cranfield-navigation-gym/log_dir/PPO_LSTM_20250321_210443_Flight_arena_model_no_fs/model/best_model.zip"

#Extra - LR exploration for camera (PPO)
# 2000000
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_model_no_fs -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 50000000 --no-frame-stack --no-static-goal -env-notes "Camera obs LR exploration " -algo-notes " "


# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_model_no_fs -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 10000000 --no-frame-stack --no-static-goal -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_model_fs -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 10000000 --frame-stack --no-static-goal -env-notes "Camera obs LR exploration " -algo-notes " "



# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO -lr 0.000003 -max-training-steps 5000000 --no-static-goal --no-frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 5000000 --no-static-goal --no-frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO -lr 0.0003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO -lr 0.003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO -lr 0.0000003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_lstm_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO_LSTM -lr 0.000003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_lstm_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_lstm_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO_LSTM -lr 0.0003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_lstm_random_goal_continuous_act_lr_framestack -obs "camera" -algo PPO_LSTM -lr 0.003 -max-training-steps 5000000 --no-static-goal --frame-stack -env-notes "Camera obs LR exploration " -algo-notes " "


# Peepeepoopoo Paris training 12/03/2025
# python3.10 train.py -run-name Flight_arena_random_pretrained_weights_random -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 5000000 --no-static-goal -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20250312_055045_Flight_arena_random/model/best_model.zip -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_random_pretrained_weights_static -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 5000000 --no-static-goal -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20250311_200659_Flight_arena_static/model/best_model.zip -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name Flight_arena_random_goal_static_robot_pretrained_random -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 3000000 --static-spawn --no-static-goal -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20250312_055045_Flight_arena_random/model/best_model.zip -env-notes "Camera obs LR exploration " -algo-notes " "


# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal -obs "camera" -algo PPO --no-frame-stack -lr 0.00003 -max-training-steps 5000000 --no-static-goal -env-notes "(random goal, no frame stack)" -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_frame_stack_random_goal -obs "camera" -algo PPO --frame-stack -lr 0.00003 -max-training-steps 5000000 --no-static-goal -env-notes "(random goal, frame stack)" -algo-notes " " 
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_lstm_random_goal -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 5000000 --no-static-goal -env-notes "(random goal, PPO LSTM)" -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name PPO_random_goal_loaded_model -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20250307_225728_ppo_test/model/best_model.zip -obs "camera" -algo PPO --frame-stack -lr 0.00003 -max-training-steps 5000000 --no-static-goal -env-notes "(random goal, loaded good statick model!)" -algo-notes " " 




# python3.10 train.py -load-model-path /home/leo/cranfield-navigation-gym/log_dir/PPO_20250305_140040_ppo_test/model/best_model.zip -run-name ppo_test -obs "camera" -algo PPO -lr 0.00003 -max-training-steps 5000000 --no-static-goal -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.0003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.03 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh

# peepee poopoo these where uncommented when we did the flight arena tests
# python3.10 train.py -run-name ppo_lstm_test -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_lstm_test -obs "camera" -algo PPO_LSTM -lr 0.0003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_lstm_test -obs "camera" -algo PPO_LSTM -lr 0.003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_lstm_test -obs "camera" -algo PPO_LSTM -lr 0.03 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# peepee poopoo end



# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.3 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.000003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.0000003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO -lr 0.00000003 -max-training-steps 5000000 -env-notes "Camera obs LR exploration " -algo-notes " "


# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test_0_0003_lr_rerun -obs "camera" -algo PPO -lr 0.0003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test_low_lr_rerun -obs "camera" -algo PPO -lr 0.003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name td3_test_0_00003 -obs "camera" -algo TD3 -lr 0.00003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name td3_test_0_0003 -obs "camera" -algo TD3 -lr 0.0003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name td3_test_0_003 -obs "camera" -algo TD3 -lr 0.003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "

# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO_LSTM -lr 0.00003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO_LSTM -lr 0.0003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "
# source /home/leo/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
# python3.10 train.py -run-name ppo_test -obs "camera" -algo PPO_LSTM -lr 0.003 -max-training-steps 1000000 -env-notes "Camera obs LR exploration " -algo-notes " "