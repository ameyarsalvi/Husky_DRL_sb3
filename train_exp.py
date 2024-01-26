

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback


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
        self.save_path = os.path.join(log_dir, "best_model_singleVS")
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


# Create log dir
import os
log_dir = "/home/asalvi/code_workspace/tmp/"
os.makedirs(log_dir, exist_ok=True)

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym

# Create environment
env = gym.make("huskyCP_gym/HuskyRL-v0",port = '23004')
env = Monitor(env, log_dir)


#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/home/asalvi/code_workspace/tmp/PPO_Husky_tensorboard/")
model = PPO("CnnPolicy", env, learning_rate=0.00184, n_steps=512, batch_size=32, n_epochs=5, gamma=0.98, gae_lambda=0.95, clip_range=0.1, verbose=1, tensorboard_log="/home/asalvi/code_workspace/tmp/PPO_Husky_tensorboard/")

# Create the callback: check every 50000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)

# Train the agent and display a progress bar
timesteps = 50000
#model.learn(total_timesteps=int(timesteps),progress_bar=True, callback=callback)
model.learn(total_timesteps=int(timesteps),progress_bar=True)
