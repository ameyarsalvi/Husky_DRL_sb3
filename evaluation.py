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

model_path = '/home/asalvi/cluster_dw/velocity_smth2/vel_p75/vel_p75.zip'

model = PPO.load(model_path, env=env, print_system_info=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25, deterministic = True)
obs = env.reset()
