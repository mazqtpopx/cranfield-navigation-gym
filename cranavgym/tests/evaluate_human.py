
import gymnasium as gym
from cranavgym.envs.drl_robot_navigation import DRLRobotNavigation

import os
import subprocess
import numpy as np

import yaml
from munch import Munch

#Pygame is necessary for the human game environment
import pygame
from queue import Queue
from threading import Thread
import cv2


#0 if you only have 1 joystick.
#1 if you use ubuntu (in my case there was a virtual controller hogging 0?)
JOYSTICK_COUNTER = 1

def load_ros_config():
    filepath = "../configs/human_configs/ros_interface_config.yaml"
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config


def load_env_config():
    filepath = "../configs/human_configs/env_config.yaml"
    with open(filepath) as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    munch_config = Munch.fromDict(config_dict)
    return munch_config




def init_pygame():
    #init pygame
    pygame.init()
    #init display
    pygame.display.init()
    screen = pygame.display.set_mode((640, 640))

    #init font
    GAME_FONT = pygame.freetype.SysFont("comicsansms", 24)

    #init joystick
    pygame.joystick.init()
    print(f"Number of joysticks: {pygame.joystick.get_count()}")
    
    #NB: the joystick number might be different
    #Hardcode the joystick number here: if only 1 joystick plugged in the 
    #number should be 0. In Ubuntu apparently there exists a default 
    #joystick which is virtual, or somethinig
    #We set a harcoded value at the top of the file, JOYSTICK_COUNTER
    

    # pygame.joystick.Joystick(JOYSTICK_COUNTER).init()
    # # Prints the joystick's name
    # JoyName = pygame.joystick.Joystick(JOYSTICK_COUNTER).get_name()
    # print(f"Name of the joystick: {JoyName}")
    # j = pygame.joystick.Joystick(JOYSTICK_COUNTER)
    # JoyAx = pygame.joystick.Joystick(JOYSTICK_COUNTER).get_numaxes()
    # print(f"Num of axis: {JoyAx}")
    return screen, GAME_FONT

def filter_deadzone(input, deadzone):
    if abs(input) < deadzone:
        return 0
    else:
        return input

def register_input_joystick():
    INVERT = True
    DEADZONE = 0.15

    

    joystick = pygame.joystick.Joystick(JOYSTICK_COUNTER)
    
    #left stick, horizontal axis
    lh = joystick.get_axis(0)
    #left stick, vertical axis
    lv = joystick.get_axis(1)
    #right stick, horizontal axis
    rh = joystick.get_axis(2)
    #right stick, vertical axis
    rv = joystick.get_axis(3)

    rt = joystick.get_axis(4)
    lt = joystick.get_axis(5)
    joy_outputs = {
        "lh": lh,
        "lv": lh,
        "rh": rh,
        "rv": rv,
    }
    print(f"{joy_outputs=}")
    # joy_init = pygame.joystick.Joystick(JOYSTICK_COUNTER).get_init()
    # print(f"joystick has been initialized: {joy_init}")

    joy_outputs = [lh, lv, rh, rv, rt, lt]
    for i in range(len(joy_outputs)):
        joy_outputs[i] = filter_deadzone(joy_outputs[i], DEADZONE)
    
    if INVERT:
        for i in range(len(joy_outputs)):
            joy_outputs[i] = -joy_outputs[i]

    return(joy_outputs)


def producer_img(buffer_img, env):
    screen, GAME_FONT = init_pygame()
    


    obs = env.reset()
    # buffer_img.put(obs)
    obs, rew, done, _, _ = env.step([0,0])
    # buffer_img.put(obs)
    reward = 0
    done = False
    info = 0
    total_reward =0
    # while done == False:
    while total_reward > -500000:
        action = 0

        #joystick
        joy_output = register_input_joystick()
        
        #Display joystick input
        text_surface_joystick, rect = GAME_FONT.render(f"joystick = {joy_output[0]:.1f}, {joy_output[1]:.1f}, {joy_output[2]:.1f}, {joy_output[3]:.1f}, {joy_output[4]:.1f}, {joy_output[5]:.1f}", (255, 255, 255))
        text_surface_reward, rect = GAME_FONT.render(f"reward = {total_reward}", (255, 255, 255))
        text_surface_done, rect = GAME_FONT.render(f"done = {done}", (255, 255, 255))
        text_surface_info, rect = GAME_FONT.render(f"info = {info}", (255, 255, 255))
        #clear the screen
        screen.fill((0,0,0))
        screen.blit(text_surface_joystick, (40, 40))
        screen.blit(text_surface_reward, (40, 80))
        screen.blit(text_surface_done, (40, 120))
        screen.blit(text_surface_info, (40, 160))

        pygame.display.flip()

        # if JOYSTICK:
        action = [0,0]
        if joy_output is not None:
            #this is the throttle
            action[0] = -(joy_output[5]-1.0)/2
            #this is turning left/right
            action[1] = -joy_output[0]
            # action[2] = process_zoom_trigger(joy_output[4],joy_output[5])
        # action = random.randint(1, 6)
        # test = env.reset()

        # print("Starting step")
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        # print("Finishing step")
        # print(f"{done=}")
        # buffer_img.put(np.zeros((160,160,3)))
        buffer_img.put(obs)
        # print("Hello")

        
        if done:
            print("We are done!")
            obs = env.reset()
            done = False

    print("We are Finishing!")



def consumer_cv2(buffer_img):
    print("init cv2 render")
    while True:
        #consume data i.e. render img
        obs = buffer_img.get()
        # print("rendering cv2")
        if obs is not None:
            #get the first 3 channels
            # print(f"{obs.shape=}")
            # img = obs[:,:,:3]
            #We have to move the axis: the obs image comes in the format [c,w,h] - we want to move to [w,h,c]
            # img = np.moveaxis(obs, 0, -1)
            img = np.float32(obs/255.0)
            # print(f"{obs.shape=}")
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    INPUT_TYPE = "joystick"
    ros_config = load_ros_config()
    env_config = load_env_config()
    buffer_img = Queue()


    env = gym.make(
        "DRLRobotNavigation-v0",
        ros_interface_config=ros_config,
        max_episode_steps=int(env_config.scenario_settings.max_steps),
        obs_space=env_config.scenario_settings.obs_space,
        reward_type="alternative",  # should put in config
        camera_noise=env_config.drl_robot_navigation.camera_noise,
        camera_noise_area_size=env_config.drl_robot_navigation.camera_noise_area_size,
        random_camera_noise_area=env_config.drl_robot_navigation.random_camera_noise_area,
        static_camera_noise_area_pos=env_config.drl_robot_navigation.static_camera_noise_area_pos,
        camera_noise_type=env_config.drl_robot_navigation.camera_noise_type,
        lidar_noise=env_config.drl_robot_navigation.lidar_noise,
        lidar_noise_area_size=env_config.drl_robot_navigation.lidar_noise_area_size,
        random_lidar_noise_area=env_config.drl_robot_navigation.random_lidar_noise_area,
        static_lidar_noise_area_pos=env_config.drl_robot_navigation.static_lidar_noise_area_pos,
        static_goal=env_config.scenario_settings.static_goal,
        static_goal_xy=env_config.scenario_settings.static_goal_xy,
        static_spawn=env_config.scenario_settings.static_spawn,
        static_spawn_xy=env_config.scenario_settings.static_spawn_xy,
    )

    # inputs = {'ros_config': ros_config, 'env_config': env_config}
    inputs = {'env': env}
    t1 = Thread(target=producer_img, args=(buffer_img,), kwargs=inputs)
    t2 = Thread(target=consumer_cv2, args=(buffer_img,))


    t1.start()
    t2.start()
    # main()
