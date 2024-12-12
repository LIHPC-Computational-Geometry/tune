from gymnasium.envs.registration import register
from environment.gymnasium_envs.trimesh_full_env.envs.trimesh import TriMeshEnvFull

register(
    id="TrimeshFull-v0",
    entry_point="environment.gymnasium_envs.trimesh_full_env.envs:TriMeshEnvFull",
    max_episode_steps=40,
    kwargs={"mesh": None, "mesh_size": 12},
)
