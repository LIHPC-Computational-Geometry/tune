from gymnasium.envs.registration import register
from environment.gymnasium_envs.trimesh_full_env.envs.trimesh import TriMeshEnvFull

register(
    id="Trimesh-v0",
    entry_point="environment.gymnasium_envs.trimesh_full_env.envs:TriMeshEnvFull",
    max_episode_steps=100,
    kwargs={"mesh": None, "mesh_size": 30, "n_darts_selected": 20, "deep": 6, "with_degree_obs": True, "action_restriction": False},
)
