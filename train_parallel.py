

import gymnasium as gym
import numpy as np
from torch import nn as nn
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage, stacked_observations, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym


# Create log dir
import os
#log_dir = "/home/asalvi/code_workspace/tmp/log0"
#os.makedirs(log_dir, exist_ok=True)

tmp_path = "/home/asalvi/code_workspace/tmp/sb3_log/log0/"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

total_timesteps = 5e6

# Callback Definitions

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model_parallel_VS")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              self.logger.record('mean_reward', mean_reward)
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True



def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        port_no = str(23004 + 2*rank)
        print(port_no)
        env = gym.make(env_id, port = port_no)
        #env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = "huskyCP_gym/HuskyRL-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
    env = VecMonitor(env, filename=tmp_path)
    env = VecTransposeImage(env, skip=False)
    
    #env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=50.0, gamma=0.99, epsilon=1e-08, norm_obs_keys=None)
    #env = VecFrameStack(env, n_stack=4) # This is an alternative to LSTM format
    #env = stacked_observations.StackedObservations(num_envs= num_cpu, n_stack = 4,observation_space =[0,255], channels_order="last")

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=tmp_path)

    checkpoint_callback = CheckpointCallback(save_freq=50000,save_path="/home/asalvi/code_workspace/tmp/sb3_log/checkpoints/",name_prefix="rl_model",save_replay_buffer=True,save_vecnormalize=True)
  # Using the followin best hyperparams
    '''
    {
    "batch_size": 128,
    "n_steps": 128,
    "gamma": 0.999,
    "learning_rate": 0.0048739625454845385,
    "ent_coef": 0.07828156338046847,
    "clip_range": 0.3,
    "n_epochs": 5,
    "gae_lambda": 0.98,
    "max_grad_norm": 0.3,
    "vf_coef": 0.5343292852148602,
    "net_arch": "medium", #But chose small
    "log_std_init": -2.224737247958312,
    "sde_sample_freq": 8,
    "activation_fn": "tanh"
}
    '''

    model = PPO("CnnPolicy", env,learning_rate=0.0001, n_steps=512, batch_size=64, n_epochs=4, ent_coef= 0.3, gamma=0.999, gae_lambda=0.98,
                clip_range=0.2, vf_coef=0.7, max_grad_norm=1.0,sde_sample_freq=8, 
                policy_kwargs=dict(normalize_images=False, log_std_init=-2.22,ortho_init=False, activation_fn=nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64])), 
                verbose=1, tensorboard_log=tmp_path)
    
    model.set_logger(new_logger)
    model.learn(total_timesteps, callback=checkpoint_callback, progress_bar= True)
    model.save("husky_VS_parallel")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
