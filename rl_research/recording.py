import typing as t

import gym
import tqdm
import numpy as np


class Agent:
    def get_action(self, obs) -> t.Any:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs) -> t.Any:
        return self.action_space.sample()


class Trajectory:
    def __init__(self):
        self.clear()

    def append(self, obs, act, obs_next, reward, done):
        self.obs.append(obs)
        self.act.append(act)
        self.obs_next.append(obs_next)
        self.reward.append(reward)
        self.done.append(done)

    def concat(self, traj):
        self.obs += traj.obs
        self.obs_next += traj.obs_next
        self.act += traj.act
        self.done += traj.done
        self.reward += traj.reward

    def clear(self):
        self.obs = []
        self.obs_next = []
        self.act = []
        self.done = []
        self.reward = []


def play_one_episode(env: t.Union[gym.Env, str], agent: Agent):
    if isinstance(env, str):
        env = gym.make(env)

    assert isinstance(env, gym.Env)

    obs = env.reset()
    info = {}
    done = False
    total_reward = 0
    total_steps = 0
    traj = Trajectory()

    while not done:
        act = agent.get_action(obs)

        obs_next, reward, done, info = env.step(act)
        traj.append(obs, act, obs_next, reward, done)

        obs = obs_next
        total_reward += reward
        total_steps  += 1

    return total_reward, total_steps, traj


def evaluate_n_episodes(env: t.Union[gym.Env, str], agent: Agent, n_episodes: int):
    epi_rewards = []
    agg_traj = Trajectory()
    for _ in tqdm.tqdm(range(n_episodes)):
        epi_reward, _, traj = play_one_episode(env, agent)
        epi_rewards.append(epi_reward)
        agg_traj.concat(traj)

    return np.mean(epi_rewards), epi_rewards, agg_traj
