#!/bin/bash
source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name test1 -algo TD3 -lr 0.03 -max-training-steps 256

source ~/cranfield-navigation-gym/cranavgym/tests/setup_env.sh
python3 train.py -run-name test2 -algo PPO -lr 0.003 -max-training-steps 256
