from gymnasium.envs.registration import register

register(
     id="huskyCP_gym/HuskyRL-v0",
     entry_point="huskyCP_gym.envs:HuskyCPEnvCone",
     max_episode_steps=1000,
)
