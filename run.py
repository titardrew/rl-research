import argparse
from pathlib import Path

import gym
import torch.nn as nn
import numpy as np

from ppo.ppo import ppo_train, Params
from ppo.model import AC_Categorical, AC_ContNormal
from ppo.vec_env import get_vec_environment


def run_ppo_hopper(run_name):

    params = Params()

    # PPO
    params.ppo.n_workers = 8
    params.ppo.traj_len = 2048
    params.ppo.use_gae = True
    params.ppo.discount_factor = 0.99
    params.ppo.gae_lambda = 0.95
    params.ppo.ppo_epoch_num = 10
    params.ppo.batch_size = 128
    params.ppo.clip_ratio = 0.2
    params.ppo.clip_value_loss = 0.2
    params.ppo.clip_grad_norm = 0.5
    params.ppo.entropy_coef = 0.0
    params.ppo.value_loss_coef = 0.5
    params.ppo.learning_rate = 2.5e-4
    params.ppo.eps = 1e-5

    # ENV:
    env_name = "HopperBulletEnv-v0"
    seed = 0xCC

    def env_fn():
        import pybullet_envs
        return gym.make(env_name)

    env = get_vec_environment(env_fn, params.ppo.n_workers)
    env.seed(seed)

    # STATS
    params.stats.freq_dump = 1
    params.stats.freq_vid = 250
    params.stats.freq_ckpt = 50
    params.stats.run_path = Path(f"runs/{run_name}")
    params.stats.ckpt_path = params.stats.run_path / "ckpt"
    params.stats.ckpt_path.mkdir(exist_ok=True, parents=True)

    # MODEL:
    shared_extractor = False
    hidden_size = 64
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    def init(module):
        #nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
        #nn.init.constant_(module.bias.data, 0)
        return module

    feature_extractor = nn.Sequential(
        init(nn.Linear(input_size, hidden_size)),
        nn.ReLU(),
        init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU()
    )
    policy_head = nn.Sequential(
        init(nn.Linear(hidden_size, action_size)), 
    )

    value_head = nn.Sequential(
        init(nn.Linear(input_size, hidden_size)),
        nn.ReLU(),
        init(nn.Linear(hidden_size, hidden_size)),
        nn.ReLU(),
        init(nn.Linear(hidden_size, 1)),
    )

    model = AC_ContNormal(
        action_size,
        feature_extractor,
        policy_head,
        value_head,
        shared_extractor
    )
    ppo_train(params, model, env, max_steps=10**7)


def run_ppo_mountain_car_cont(run_name):
    params = Params()

    # PPO
    params.ppo.n_workers = 16
    params.ppo.traj_len = 256
    params.ppo.use_gae = False
    params.ppo.discount_factor = .99
    params.ppo.gae_lambda = .95
    params.ppo.ppo_epoch_num = 4
    params.ppo.batch_size = 16
    params.ppo.clip_ratio = .2
    params.ppo.clip_value_loss = .2
    params.ppo.clip_grad_norm = .5
    params.ppo.entropy_coef = .01
    params.ppo.value_loss_coef = .5
    params.ppo.learning_rate = 7e-4
    params.ppo.eps = 1e-5

    # ENV:
    env_name = "MountainCarContinuous-v0"
    seed = 0xCC

    def env_fn(): return gym.make(env_name)
    env = get_vec_environment(env_fn, params.ppo.n_workers)
    env.seed(seed)

    # STATS
    params.stats.freq_dump = 25
    params.stats.freq_vid = 250
    params.stats.freq_ckpt = 50
    params.stats.run_path = Path(f"runs/{run_name}")
    params.stats.ckpt_path = params.stats.run_path / "ckpt"
    params.stats.ckpt_path.mkdir(exist_ok=True, parents=True)

    # MODEL:
    shared_extractor = True
    hidden_size = 128
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    feature_extractor = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU()
    )
    policy_head = nn.Sequential(
        nn.Linear(hidden_size, action_size), 
    )

    value_head = nn.Sequential(
        nn.Linear(hidden_size, 1),
    )

    model = AC_ContNormal(
        action_size,
        feature_extractor,
        policy_head,
        value_head,
        shared_extractor
    )
    ppo_train(params, model, env, max_updates=5000)

def run_ppo_carpole(run_name):
    params = Params()

    # PPO
    params.ppo.n_workers = 8
    params.ppo.traj_len = 256
    params.ppo.use_gae = True
    params.ppo.discount_factor = .99
    params.ppo.gae_lambda = .95
    params.ppo.ppo_epoch_num = 4
    params.ppo.batch_size = 16
    params.ppo.clip_ratio = .2
    params.ppo.clip_value_loss = .2
    params.ppo.clip_grad_norm = .5
    params.ppo.entropy_coef = .01
    params.ppo.value_loss_coef = .5
    params.ppo.learning_rate = 7e-4
    params.ppo.eps = 1e-5

    # ENV:
    env_name = "CartPole-v1"
    seed = 0xCC

    def env_fn(): return gym.make(env_name)
    env = get_vec_environment(env_fn, params.ppo.n_workers)
    env.seed(seed)

    # STATS
    params.stats.freq_dump = 25
    params.stats.freq_vid = 250
    params.stats.freq_ckpt = 50
    params.stats.run_path = Path(f"runs/{run_name}")
    params.stats.ckpt_path = params.stats.run_path / "ckpt"
    params.stats.ckpt_path.mkdir(exist_ok=True, parents=True)

    # MODEL:
    shared_extractor = True
    hidden_size = 128
    input_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    feature_extractor = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU()
    )
    policy_head = nn.Sequential(
        nn.Linear(hidden_size, action_size), 
    )

    value_head = nn.Sequential(
        nn.Linear(hidden_size, 1),
    )

    model = AC_Categorical(
        feature_extractor,
        policy_head,
        value_head,
        shared_extractor
    )
    ppo_train(params, model, env, max_updates=275)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-name", type=str)
    args = parser.parse_args()

    run_ppo_hopper(args.run_name)
    #run_ppo_mountain_car_cont(args.run_name)
    #run_ppo_carpole(args.run_name)
