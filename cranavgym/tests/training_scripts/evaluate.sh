#!/bin/bash

#PPO trained on 0x0, evaluated on default map
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py --evaluate-only -load-model-path ~/cranfield-navigation-gym/log_dir/PPO_20240822_204128_camera_static_goal/model/best_model.zip -run-name default_baseline_default_map -obs camera -algo PPO --static-goal -env-notes "Camera obs, evaluation " -algo-notes "PPO cam "
