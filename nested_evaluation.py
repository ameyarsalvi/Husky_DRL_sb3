import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym

import os
eval_log_dir = "/home/asalvi/code_workspace/tmp/eval/"
os.makedirs(eval_log_dir, exist_ok=True)

#[0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for x in [0.8, 0.85, 0.9, 0.95]:
    env = gym.make("huskyCP_gym/HuskyRL-v0",port=23002,seed=1,track_vel = x)
    env = Monitor(env, eval_log_dir)
    abs_path = '/home/asalvi/cluster_dw/velocity_smth2/'
    specifier = 'vel_p'+ str(int(x*100))
    model_path = abs_path + specifier + '/' + specifier
    model = PPO.load(model_path, env=env, print_system_info=True)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25, deterministic = True)
    obs = env.reset()
    print('finished test'+str(x))