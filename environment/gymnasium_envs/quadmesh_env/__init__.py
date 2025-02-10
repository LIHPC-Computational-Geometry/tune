from gymnasium.envs.registration import register
from environment.gymnasium_envs.quadmesh_env.envs.quadmesh import QuadMeshEnv

register(
    id="Quadmesh-v0",
    entry_point="environment.gymnasium_envs.quadmesh_env.envs:QuadMeshEnv",
    max_episode_steps=100,
    kwargs={"mesh": None, "mesh_size": 30, "n_darts_selected": 20, "deep": 6, "with_degree_obs": True, "action_restriction": False},
)
