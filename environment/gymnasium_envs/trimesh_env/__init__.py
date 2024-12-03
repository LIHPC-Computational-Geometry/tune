from gymnasium.envs.registration import register
from environment.gymnasium_envs.trimesh_env.envs.trimesh import TriMeshEnv
import gymnasium as gym

register(
    id="trimesh-v0",
    entry_point="environment.gymnasium_envs.trimesh_env.envs:TriMeshEnv",
    max_episode_steps=100,
)
