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

'''
class GetEnvVar(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        #self.training_env = env

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.get_attr('self.log_err_feat')
        #self.logger.record("random_value", value)
        print(value)
        return value
'''   

# Create environment
env = gym.make("huskyCP_gym/HuskyRL-v0",port=23002,seed=1,track_vel = 0.75)
env = Monitor(env, eval_log_dir)
#value = env.unwrapped.get_attr('self.log_err_feat')
#print(value)

#model_path = "/home/asalvi/Downloads/velocity/vel_p65.zip"
model_path = '/home/asalvi/Downloads/velocity/vel_p95.zip'
# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True'
# to compare the system on which the model was trained vs the current one
#model = PPO.load("/home/asalvi/code_workspace/tmp/sb3_log/log2/logHS/best_model_parallel_VS.zip", env=env, print_system_info=True)
model = PPO.load(model_path, env=env, print_system_info=True)

# '/home/asalvi/Downloads/feb3/single_32/HuskyVSsingle32.zip'

#callback = GetEnvVar()
#value = callback._on_step()
#print(value)


# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic = True)
obs = env.reset()
