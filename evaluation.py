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

# Create environment
env = gym.make("huskyCP_gym/HuskyRL-v0",port=23002)
env = Monitor(env, eval_log_dir)

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
#model = PPO.load("/home/asalvi/code_workspace/tmp/sb3_log/log2/logHS/best_model_parallel_VS.zip", env=env, print_system_info=True)
model = PPO.load("/home/asalvi/Downloads/logs_br/log_ten/best_model_parallel_VS.zip", env=env, print_system_info=True)


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10000, deterministic = False)
