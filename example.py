import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

from openrl.configs.config import create_config_parser
from openrl.modules.common.ppo_net import PPONet as Net
from openrl.runners.common.ppo_agent import PPOAgent as Agent
from leggedgym2openrl import make_env

def train():
    env = make_env("anymal_c_rough")
    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    
    net = Net(
        env,
        cfg=cfg,
    )
    agent = Agent(net)
    # start training, set total number of training steps to 20000
    agent.train(total_time_steps=40000)

    # begin to test
    # The trained agent sets up the interactive environment it needs.
    agent.set_env(env)
    # Initialize the environment and get initial observations and environmental information.
    obs = env.reset()
    done = False
    step = 0
    total_re = 0.0
    while not np.any(done):
        # Based on environmental observation input, predict next action.
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
        step += 1
        if step % 50 == 0:
            print(f"{step}: reward:{np.mean(r)}")
        total_re += np.mean(r)
    print(f"Total reward:{total_re}")

if __name__ == '__main__':
    train()
