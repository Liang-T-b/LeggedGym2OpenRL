from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

import isaacgym
from legged_gym.envs import Anymal
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.envs import *
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.utils import get_args, task_registry


def make_env(name, args=None, env_cfg=None):
    # if no args passed get command line arguments
    if args is None:
        args = get_args()
    # check if there is a registered env with that name
    if name in task_registry.task_classes:
        task_class = task_registry.get_task_class(name)
    else:
        raise ValueError(f"Task with name: {name} was not registered")
    if env_cfg is None:
        # load config files
        env_cfg, _ = task_registry.get_cfgs(name)
    # override cfg from args (if specified)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    set_seed(env_cfg.seed)
    # parse sim params (convert to dict first)
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = task_class(   cfg=env_cfg,
                        sim_params=sim_params,
                        physics_engine=args.physics_engine,
                        sim_device=args.sim_device,
                        headless=args.headless)
    
    return LeggedGym2OpenRL(env)


class LeggedGym2OpenRL:
    def __init__(self, env: Anymal) -> Anymal:
        self.env = env

        high_obs = np.full(self.env.cfg.env.num_observations, self.env.cfg.normalization.clip_observations)
        self.obs_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        high_ac = np.full(self.env.cfg.env.num_actions, self.env.cfg.normalization.clip_actions)
        self.ac_space = spaces.Box(-high_ac, high_ac, dtype=np.float32)

    @property
    def parallel_env_num(self) -> int:
        return self.env.cfg.env.num_envs

    @property
    def action_space(
        self,
    ) -> Union[spaces.Space[ActType], spaces.Space[WrapperActType]]:
        """Return the :attr:`Env` :attr:`action_space` unless overwritten then the wrapper :attr:`action_space` is used."""
        return self.ac_space

    @property
    def observation_space(
        self,
    ) -> Union[spaces.Space[ObsType], spaces.Space[WrapperObsType]]:
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        return self.obs_space

    def reset(self, **kwargs):
        """Reset all environments."""
        obs, _ = self.env.reset()
        return obs.unsqueeze(1).cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""

        actions = torch.from_numpy(actions).squeeze(1)

        obs_buff, _, self._rew, self._resets, self._extras = self.env.step(actions)

        obs = obs_buff.unsqueeze(1).cpu().numpy()
        rewards = self._rew.cpu().numpy()
        dones = self._resets.cpu().numpy().astype(bool)

        infos = []
        for i in range(dones.shape[0]):
            infos.append({})

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def agent_num(self):
        return 1

    @property
    def use_monitor(self):
        return False

    @property
    def env_name(self):
        return "Isaac-" + self.env.cfg.env.task_name

    def batch_rewards(self, buffer):
        return {}