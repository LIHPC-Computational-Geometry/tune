from __future__ import annotations

import random
import torch
import yaml
import os
import time
import wandb

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement, ProgressBarCallback

from wandb.integration.sb3 import WandbCallback

from environment.gymnasium_envs import quadmesh_env
import mesh_model.random_quadmesh as QM
from mesh_model.reader import read_gmsh
from view.mesh_plotter.mesh_plots import dataset_plt
from training.exploit_SB3_policy import testPolicy


def make_env(config, training_mesh):
    """
    Function to create gym environment for vectorize learning.
    :param config: configuration yaml file
    :param training_mesh: mesh to learn on
    :return: gymnasium environment
    """
    def _init():
        return gym.make(
            config["env"]["env_id"],
            mesh=training_mesh,
            max_episode_steps=config["env"]["max_episode_steps"],
            n_darts_selected=config["env"]["n_darts_selected"],
            deep=config["env"]["deep"],
            action_restriction=config["env"]["action_restriction"],
            with_degree_obs=config["env"]["with_degree_observation"],
        )
    return _init

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "experiment": experiment_name,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": config["ppo"]["batch_size"],
            "epochs": config["ppo"]["n_epochs"],
            "clip_range": config["ppo"]["clip_range"],
            "training_meshes": config["dataset"]["training_mesh_file_path"],
            "evaluation_meshes": config["dataset"]["evaluation_mesh_file_path"],
            "max_steps": config["env"]["max_episode_steps"],
            "max_timesteps": config["total_timesteps"],
            "deep": config["env"]["deep"],
            "with_degree": config["env"]["with_degree_observation"],
            "nb_darts_selected": config["env"]["n_darts_selected"],
            "reward_mode": config["env"]["reward_function"],


        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "normalized_return": 0,
            "rollout/ep_len_mean": 0.0,
            "rollout/ep_rew_mean": 0.0
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, model, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.episode_rewards = []
        self.mesh_reward = 0
        self.current_episode_reward = 0
        self.episode_count = 0
        self.current_episode_length = 0
        self.actions_info = {
            "episode_valid_actions": 0,
            "episode_invalid_topo": 0,
            "episode_invalid_geo": 0,
            "nb_flip_cw" : 0,
            "nb_flip_cntcw": 0,
            "nb_split": 0,
            "nb_collapse": 0,
            "nb_cleanup": 0,
            "nb_invalid_flip": 0,
            "nb_invalid_split": 0,
            "nb_invalid_collapse": 0,
            "nb_invalid_cleanup": 0,
        }
        self.final_distance = 0
        self.normalized_return = 0


    def _on_step(self) -> bool:
        """
        Record different learning variables to monitor
        """
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        self.actions_info["episode_valid_actions"] += self.locals["infos"][0].get("valid_action", 0.0)
        self.actions_info["episode_invalid_topo"] += self.locals["infos"][0].get("invalid_topo", 0.0)
        self.actions_info["episode_invalid_geo"] += self.locals["infos"][0].get("invalid_geo", 0.0)
        self.actions_info["nb_flip_cw"] += self.locals["infos"][0].get("flip_cw", 0.0)
        self.actions_info["nb_flip_cntcw"] += self.locals["infos"][0].get("flip_cntcw", 0.0)
        self.actions_info["nb_split"] += self.locals["infos"][0].get("split", 0.0)
        self.actions_info["nb_collapse"] += self.locals["infos"][0].get("collapse", 0.0)
        self.actions_info["nb_cleanup"] += self.locals["infos"][0].get("cleanup", 0.0)
        self.actions_info["nb_invalid_flip"] += self.locals["infos"][0].get("invalid_flip", 0.0)
        self.actions_info["nb_invalid_split"] += self.locals["infos"][0].get("invalid_split", 0.0)
        self.actions_info["nb_invalid_collapse"] += self.locals["infos"][0].get("invalid_collapse", 0.0)
        self.actions_info["nb_invalid_cleanup"] += self.locals["infos"][0].get("invalid_cleanup", 0.0)

        self.mesh_reward += self.locals["infos"][0].get("mesh_reward", 0.0)

        # When the episode is over
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward) # global rewards obtained during the episode
            mesh_ideal_reward = self.locals["infos"][0].get("mesh_ideal_rewards", 0.0) # maximum achievable reward
            if mesh_ideal_reward > 0:
                self.normalized_return = self.mesh_reward/ mesh_ideal_reward
            else:
                self.normalized_return = 0

            self.final_distance = self.locals["infos"][0].get("distance", 0.0)
            self.logger.record("final_distance", self.final_distance)
            self.logger.record("valid_actions", self.actions_info["episode_valid_actions"]*100/self.current_episode_length if self.current_episode_length > 0 else 0)
            self.logger.record("actions/n_invalid_topo", self.actions_info["episode_invalid_topo"]*100/(self.current_episode_length - self.actions_info["episode_valid_actions"]) if self.actions_info["episode_invalid_topo"] > 0 else 0)
            self.logger.record("actions/n_invalid_geo", self.actions_info["episode_invalid_geo"]*100/(self.current_episode_length - self.actions_info["episode_valid_actions"]) if self.actions_info["episode_invalid_geo"] > 0 else 0)
            self.logger.record("actions/nb_flip_cw", self.actions_info["nb_flip_cw"])
            self.logger.record("actions/nb_flip_cntcw", self.actions_info["nb_flip_cntcw"])
            self.logger.record("actions/nb_split", self.actions_info["nb_split"])
            self.logger.record("actions/nb_collapse", self.actions_info["nb_collapse"])
            self.logger.record("actions/nb_cleanup", self.actions_info["nb_cleanup"])
            self.logger.record("actions/invalid_flip", self.actions_info["nb_invalid_flip"]*100/(self.actions_info["nb_flip_cw"]+self.actions_info["nb_flip_cntcw"]) if (self.actions_info["nb_flip_cw"]+self.actions_info["nb_flip_cntcw"]) > 0 else 0)
            self.logger.record("actions/invalid_split", self.actions_info["nb_invalid_split"]*100/self.actions_info["nb_split"] if self.actions_info["nb_split"] > 0 else 0)
            self.logger.record("actions/invalid_collapse", self.actions_info["nb_invalid_collapse"]*100/self.actions_info["nb_collapse"]if self.actions_info["nb_collapse"] > 0 else 0)
            self.logger.record("actions/invalid_cleanup", self.actions_info["nb_invalid_cleanup"]*100/self.actions_info["nb_cleanup"]if self.actions_info["nb_cleanup"] > 0 else 0)
            self.logger.record("episode_mesh_reward", self.mesh_reward)
            self.logger.record("episode_reward", self.current_episode_reward)
            self.logger.record("normalized_return", self.normalized_return)
            self.logger.record("episode_length", self.current_episode_length)

            is_success = self.locals["infos"][0].get("is_success", 0.0)  # Default value: 0.0
            self.logger.record("episode_success", is_success)

            self.logger.dump(step=self.episode_count)
            self.current_episode_reward = 0  #  resets global episode reward
            self.mesh_reward = 0 #  resets mesh episode reward
            self.current_episode_length = 0
            #reset actions info
            for key in self.actions_info.keys():
                self.actions_info[key] = 0
            self.episode_count += 1  # Increment episode counter

        return True

    def _on_training_end(self) -> None:
        """
        Records policy evaluation results : before and after dataset images
        Save registry counts of observation in a csv file. Records analysis
        """

        obs_registry = self.locals["infos"][0].get("observation_registry", None)
        if obs_registry is not None:
            filename = "training/observation_counts_" + experiment_name + ".csv"

            obs_registry.save(filename)
            print(f"Counts saved at {filename}")

            counts = obs_registry.df["counts"]

            self.logger.record("observation/n_observation", len(counts))
            self.logger.record("observation/mean", np.mean(counts))
            self.logger.record("observation/median", np.median(counts))
            self.logger.record("observation/min", np.min(counts))
            self.logger.record("observation/max", np.max(counts))

            values = counts.values
            values = np.sort(values)[::-1] #Descending sort
            max_values = 300
            if len(values) > max_values:
                values =values[:max_values] # We keep only 500 values for histogram

            figure, ax = plt.subplots()
            ax.bar(range(len(values)),values)
            ax.set_title("Observation counts")
            # Close the figure after logging it
            self.logger.record("observation/counts", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            self.logger.dump(step=self.num_timesteps)

        #Test policy
        print("-------- Testing policy ----------")
        #mesh = read_gmsh("mesh_files/medium_quad.msh")
        dataset = [QM.random_mesh() for _ in range(2)] # dataset of 9 meshes of size 30
        before = dataset_plt(dataset) # plot the datasat as image
        length, wins, rewards, normalized_return, final_meshes = testPolicy(self.model, 1, config, dataset) # test model policy on the dataset
        after = dataset_plt(final_meshes)
        self.logger.record("figures/before", Figure(before, close=True), exclude=("stdout", "log"))
        self.logger.record("figures/after", Figure(after, close=True), exclude=("stdout", "log"))
        self.logger.dump(step=self.num_timesteps)

if __name__ == '__main__':

    # PARAMETERS CONFIGURATION
    with open("../training/config/quadmesh_config_PPO_SB3.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config["experiment_name"]

    # SEEDING
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # WANDB
    run = wandb.init(
        project="Quadmesh-learning",
        name=config["experiment_name"],
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    # EVALUATION CALLBACKS

    # Separate evaluation env
    eval_env = Monitor(gym.make(
        config["eval"]["eval_env_id"],
        mesh = read_gmsh(config["dataset"]["evaluation_mesh_file_path"]),
        max_episode_steps=config["eval"]["max_episode_steps"],
        n_darts_selected=config["env"]["n_darts_selected"],
        deep= config["env"]["deep"],
        action_restriction=config["env"]["action_restriction"],
        with_degree_obs=config["env"]["with_degree_observation"],
        render_mode=config["env"]["render_mode"],
    ))
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=config["eval"]["max_no_improvement_evals"],
        min_evals=config["eval"]["min_evals"],
        verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=config["eval"]["eval_freq"], callback_after_eval=stop_train_callback, verbose=1)

    # Create tensorboard log dir
    log_dir = config["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    training_mesh = read_gmsh(config["dataset"]["training_mesh_file_path"])

    # Create the environment

    if config["env"]["n_vec_envs"]>0:
        env = SubprocVecEnv([make_env(config, training_mesh) for _ in range(config["env"]["n_vec_envs"])])
    else:
        env = gym.make(
            config["env"]["env_id"],
            mesh=training_mesh,
            max_episode_steps=config["env"]["max_episode_steps"],
            n_darts_selected=config["env"]["n_darts_selected"],
            deep=config["env"]["deep"],
            action_restriction=config["env"]["action_restriction"],
            with_degree_obs=config["env"]["with_degree_observation"],
            render_mode=config["env"]["render_mode"],
            obs_count=config["env"]["obs_count"],
        )
        check_env(env, warn=True)


    model = PPO(
        policy=config["ppo"]["policy"],
        env=env,
        n_steps=config["ppo"]["n_steps"],
        n_epochs=config["ppo"]["n_epochs"],
        batch_size=config["ppo"]["batch_size"],
        learning_rate=config["ppo"]["learning_rate"],
        gamma=config["ppo"]["gamma"],
        verbose=1,
        tensorboard_log=log_dir
    )

    start_time = time.perf_counter()
    print("-----------Starting learning-----------")
    model.learn(
        total_timesteps=config["total_timesteps"],
        tb_log_name=config["experiment_name"],
        callback=[HParamCallback(), WandbCallback(model_save_path=config["paths"]["wandb_model_saving_dir"]+config["experiment_name"]), TensorboardCallback(model)],
        progress_bar=True
    )
    end_time = time.perf_counter()
    print("-----------Learning ended------------")
    print(f"Temps d'apprentissage : {end_time - start_time:.4} s")
    model.save(config["paths"]["policy_saving_dir"]+config["experiment_name"])
    run.finish()

