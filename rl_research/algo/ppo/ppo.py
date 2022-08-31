from typing import Optional as Opt
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import av
import gym
from gym.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn

from rl_research.algo.ppo.rollout import TorchStorage
from rl_research.algo.ppo.statistics import Statistics, Timer
from rl_research.algo.ppo.running_mean_std import RunningMeanStd
from rl_research.algo.ppo.model import AC_ContNormal
import rl_research.recording as recording


@dataclass
class Preprocessor_State:
    obs_shape:       tuple       = (1,)
    obs_mean:        np.ndarray  = field(default=np.array([0.0]))
    obs_var:         np.ndarray  = field(default=np.array([1.0]))
    return_mean:     float       = 0.0
    return_var:      float       = 1.0
    obs_clip:        np.ndarray  = field(default=np.array([float("inf")]))
    reward_clip:     float       = float("inf")
    discount_factor: float       = 0.99


class Preprocessor:
    EPS_ = 1e-7

    def __init__(self,
        obs_shape:       tuple,
        obs_clip:        float,
        reward_clip:     float,
        discount_factor: float
    ):
        self.obs_shape = obs_shape
        self.discount_factor = discount_factor

        self.obs_clip = obs_clip
        self.reward_clip = reward_clip

        self.running_mean_std_obs    = RunningMeanStd(shape=self.obs_shape)
        self.running_mean_std_return = RunningMeanStd()

        self.return_vec = 0.0  # Supposed to have a shape = (batch_size, )

    def process(self, obs_vec: np.ndarray, reward_vec: Opt[np.ndarray]):
        obs_mean = self.running_mean_std_obs.mean
        obs_var  = self.running_mean_std_obs.var

        obs_vec_    = (obs_vec - obs_mean) / np.sqrt(obs_var + self.EPS_)
        obs_vec_    = np.clip(obs_vec_, -self.obs_clip, self.obs_clip)

        if reward_vec is not None:
            ret_var  = self.running_mean_std_return.var

            reward_vec_ = reward_vec / np.sqrt(ret_var + self.EPS_)
            reward_vec_ = np.clip(reward_vec_, -self.reward_clip, self.reward_clip)
        else:
            reward_vec_ = None

        return obs_vec_, reward_vec_

    def update(self, obs_vec: np.ndarray, reward_vec: Opt[np.ndarray], done_vec: np.ndarray):
        self.running_mean_std_obs.update(obs_vec)

        if reward_vec is not None:
            self.return_vec = self.return_vec * self.discount_factor + reward_vec
            self.running_mean_std_return.update(self.return_vec)

            self.return_vec[done_vec] = 0.0


    def update_and_process(self, obs_vec: np.ndarray, reward_vec: np.ndarray, done_vec: np.ndarray):
        self.update(obs_vec, reward_vec, done_vec)
        return self.process(obs_vec, reward_vec)

    def get_state(self):
        return Preprocessor_State(
            obs_shape       = self.obs_shape,
            obs_mean        = self.running_mean_std_obs.mean,
            obs_var         = self.running_mean_std_obs.var,
            return_mean     = self.running_mean_std_return.mean,
            return_var      = self.running_mean_std_return.var,
            obs_clip        = self.obs_clip,
            reward_clip     = self.reward_clip,
            discount_factor = self.discount_factor,
        )

    @staticmethod
    def from_state(state: Preprocessor_State):
        prep = Preprocessor(state.obs_shape, state.obs_clip, state.reward_clip, state.discount_factor)
        prep.set_state(state)
        return prep

    def set_state(self, state: Preprocessor_State):
        self.obs_shape                    = self.obs_shape,
        self.running_mean_std_obs.mean    = state.obs_mean
        self.running_mean_std_obs.var     = state.obs_var
        self.running_mean_std_return.mean = state.return_mean
        self.running_mean_std_return.var  = state.return_var
        self.obs_clip                     = state.obs_clip
        self.reward_clip                  = state.reward_clip
        self.discount_factor              = state.discount_factor


def ppo_update(
    params,
    model=None,
    opt=None
):
    assert not (model is None and opt is None), \
        "Either model or opt has to be specified"
    if opt is None:
        opt = torch.optim.Adam(
            model.parameters, lr=params.learning_rate, eps=params.eps)

    def _update_fn(model, storage, statistics):

        adv = storage.get_normalized_advantage(
            params.use_gae,
            params.discount_factor,
            params.gae_lambda
        )

        mean_value_loss  = 0
        mean_policy_loss = 0
        mean_entropy     = 0
        mean_loss        = 0
        mean_clip_frac   = 0
        mean_approx_kl   = 0
        mean_action_diff = 0

        batch_generator = storage.generate_fwd

        for epoch in range(params.ppo_epoch_num):
            for batch in batch_generator(adv, params.batch_size):
                (
                    obs_prev,
                    h_states,
                    actions,
                    value_preds_saved,
                    returns,
                    not_dones,
                    log_pi_saved,
                    adv_target,
                ) = batch


                policy_out, value_pred = model.policy_and_value(obs_prev)
                act, log_pi = model.select_action(policy_out, action=actions)
                entropy = model.entropy(policy_out).mean()

                r = (log_pi.unsqueeze(-1) - log_pi_saved).exp()
                r1 = adv_target * r
                r2 = adv_target * r.clamp(1.0 - params.clip_ratio,
                                          1.0 + params.clip_ratio)
                policy_loss = -torch.min(r1, r2).mean()

                if params.clip_value_loss:
                    value_pred_clipped = (
                        value_preds_saved +
                        torch.clamp(
                            value_pred - value_preds_saved,
                            -params.clip_value_loss,
                            +params.clip_value_loss
                        )
                    )
                    v_loss = torch.pow(value_pred - returns, 2)
                    v_loss_clip = torch.pow(value_pred_clipped - returns, 2)
                    value_loss = torch.max(v_loss, v_loss_clip).mean() * 0.5
                else:
                    value_loss = torch.pow(
                        value_pred - returns, 2).mean() * 0.5

                loss = value_loss * params.value_loss_coef + policy_loss
                reg = entropy * params.entropy_coef
                with torch.no_grad():
                    clip_frac = torch.gt(
                        (r - 1.0).abs(), params.clip_ratio).float().mean()
                    approx_kl = (
                        log_pi.unsqueeze(-1) - log_pi_saved).pow(2).mean() * 0.5
                    action_diff = (act - actions).pow(2).mean()

                opt.zero_grad()
                (loss - reg).backward()
                nn.utils.clip_grad_norm_(model.parameters, params.clip_grad_norm)
                opt.step()

                mean_value_loss  += value_loss.item()
                mean_policy_loss += policy_loss.item()
                mean_entropy     += entropy.item()
                mean_loss        += loss.item()
                mean_clip_frac   += clip_frac.item()
                mean_approx_kl   += approx_kl.item()
                mean_action_diff += action_diff.item()

        n_loop_iter = (
            params.ppo_epoch_num *
            params.traj_len *
            params.n_workers //
            params.batch_size
        )

        mean_value_loss  /= n_loop_iter
        mean_policy_loss /= n_loop_iter
        mean_entropy     /= n_loop_iter
        mean_loss        /= n_loop_iter
        mean_clip_frac   /= n_loop_iter
        mean_approx_kl   /= n_loop_iter
        mean_action_diff /= n_loop_iter

        mean_log_pi = storage._st["log_pi"].mean().item()
        mean_v_pred = storage._st["value_pred"].mean().item()
        mean_ret    = storage._st["returns"].mean().item()
        mean_adv    = adv.mean().item()

        statistics.add_agent_data({
            "loss":        mean_loss,
            "value_loss":  mean_value_loss,
            "policy_loss": mean_policy_loss,
            "entropy":     mean_entropy,
            "clip_frac":   mean_clip_frac,
            "approx_kl":   mean_approx_kl,
            "log_pi":      mean_log_pi,
            "value_pred":  mean_v_pred,
            "returns":     mean_ret,
            "advantages":  mean_adv,
            "logstd":      model.logvar.mean().item(),
            "act_diff":    mean_action_diff,
        })

    return _update_fn


def save_video(frames, name, fps=15):
    container = av.open(str(name), mode="w")
    stream = container.add_stream("mpeg4", rate=fps)
    shape = frames[0].shape
    dtype = frames[0].dtype
    if len(shape) == 2:
        fmt = "gray"
    elif len(shape) == 3:
        fmt = "rgb24"
    else:
        raise ValueError("Frame may have 2 or 3 "
                        f"dimensions, not {len(shape)}")

    if dtype != "uint8":
        warnings.warn(f"Frame dtype has to be uint8, not {dtype}. "
                       "It will be min-max normalized and casted.")
        def norm_uint8(x):
            x = x - x.min()
            x /= x.max()
            x *= 255
            return x.astype(np.uint8)

        norm_fn = norm_uint8
    else:
        norm_fn = lambda x: x

    stream.width = shape[1]
    stream.height = shape[0]
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        frame = av.VideoFrame.from_ndarray(norm_fn(frame), format=fmt)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@dataclass
class Params_PPO:
    n_workers:       int   = 1
    traj_len:        int   = 256
    use_gae:         bool  = True
    discount_factor: float = .99
    gae_lambda:      float = .95
    ppo_epoch_num:   int   = 4
    batch_size:      int   = 15
    clip_ratio:      float = .2
    clip_value_loss: float = .2
    clip_grad_norm:  float = .5
    entropy_coef:    float = .01
    value_loss_coef: float = .5
    learning_rate:   float = 7e-4
    eps:             float = 1e-5
    obs_clip:        float = float("inf")
    reward_clip:     float = float("inf")


@dataclass
class Params_Stats:
    run_path:  Path = field(default=Path())
    ckpt_path: Path = field(default=Path())

    freq_dump: int = 10
    freq_vid:  int = 50
    freq_ckpt: int = 50


@dataclass
class Params:
    ppo:   Params_PPO   = field(default_factory=Params_PPO)
    stats: Params_Stats = field(default_factory=Params_Stats)


def ppo_train(
    params,
    model,
    env,
    max_episodes=None,
    max_updates=None,
    max_steps=None
):

    assert not (max_episodes is None and
                max_updates is None and
                max_steps is None), \
        "Either max_episodes or max_updates has to be specified"
    max_episodes = max_episodes or float("inf")
    max_updates = max_updates or float("inf")
    max_steps = max_steps or float("inf")

    storage = TorchStorage(
        params.ppo.traj_len,
        params.ppo.n_workers,
        env.observation_space.shape,
        env.action_space,
        1
    )

    statistics = Statistics(
        params.ppo.n_workers,
        tb_run_experiment=params.stats.run_path / "tb",
    )

    update_fn = ppo_update(
        params.ppo,
        model=model
    )

    preprocessor = Preprocessor(
            env.observation_space.shape,
            params.ppo.obs_clip,
            params.ppo.reward_clip,
            params.ppo.discount_factor)

    obs_cur_raw = env.reset_all()
    done_cur = [False for _ in range(params.ppo.n_workers)]
    obs_cur, _ = preprocessor.update_and_process(
            np.asarray(obs_cur_raw), None, np.asarray(done_cur))

    n_updates = 0

    best_metric = -float("inf")

    total_timer = Timer()
    perf_timer = Timer()

    total_timer.start()
    while (env.n_episodes < max_episodes
           and n_updates < max_updates
           and statistics.total_steps < max_steps):
        # TODO: rnn state passing to network
        with torch.no_grad():
            policy_out, value_pred = model.policy_and_value(
                torch.FloatTensor(obs_cur))
            act, log_pi_chosen = model.select_action(policy_out)
        obs_prev = obs_cur
        done_prev = done_cur

        ### TODO: is_good mask (bad_transition)
        #         Rollout layout:
        # o0 -> o1 -> o2 -> o3 -> o4 -| o0
        # o0 -> o1 -> o2 -> o3 -| o0 -> o1
        # o0 -> o1 -> o2 -> o3 -> o4 -| o0
        # o0 -> o1 -| o0 -> o1 -> o2 -> o3

        if isinstance(env.action_space, Box):
            act_clipped = np.clip(act.cpu().numpy(),
                          env.action_space.low,
                          env.action_space.high)
        else:
            act_clipped = act.cpu().numpy()

        perf_timer.start()
        obs_cur_raw, reward_raw, done_cur, info = env.step_nonstop(act_clipped)

        obs_cur, reward = preprocessor.update_and_process(np.asarray(obs_cur_raw), np.asarray(reward_raw), np.asarray(done_cur))

        env_time = perf_timer.get_elapsed_seconds()

        statistics.add_env_data({
            "time_per_env": env_time,
            "actions":      act,
            "reward":       reward_raw,
            "done":         done_prev,
            "info":         info,
            "obs":          obs_prev,
        })

        if isinstance(env.action_space, Discrete):
            act = torch.LongTensor(act)
            act.unsqueeze_(-1)
        else:
            act = torch.FloatTensor(act)

        storage.save_rollout({
            "obs":        torch.FloatTensor(obs_prev),
            "value_pred": torch.FloatTensor(value_pred),
            "log_pi":     torch.FloatTensor(log_pi_chosen).unsqueeze(-1),
            "rewards":    torch.FloatTensor(reward).unsqueeze(-1),
            "actions":    act,
            "dones":      torch.FloatTensor(done_prev).unsqueeze(-1),
        })

        if storage.i_step == storage.traj_len - 1:

            with torch.no_grad():
                value_pred = model.value(torch.FloatTensor(obs_cur))
            storage.finalize_storage(
                value_pred.view(-1, 1),
                torch.FloatTensor(done_cur).view(-1, 1),
            )
            update_fn(model, storage, statistics)
            n_updates += 1

            if n_updates % params.stats.freq_dump == 0:
                statistics.reduce(total_timer.get_elapsed_seconds())
                statistics.dump()

            if n_updates % params.stats.freq_ckpt == 0:
                path = params.stats.ckpt_path
                if statistics.stats["reward/epi_mean"] > best_metric:
                    model.save(path / "best.pth")
                    torch.save(preprocessor.get_state(), path / "best.preproc.pth")
                model.save(path / f"{n_updates}.pth")
                torch.save(preprocessor.get_state(), path / f"{n_updates}.preproc.pth")

            if n_updates % params.stats.freq_vid == 0:
                # import pdb; pdb.set_trace()
                has_started = env.start_recording(n_episodes=1, size=300, mode="rgb_array")
                if has_started:
                    print("Started recording video...")

            res = env.try_get_recordings()
            if res:
                print("Uploading video recordings to tensorboard...")
                statistics.dump_video(res, "episodes", fps=15, size=None)
            # save_video(res[0], params.stats.run_path / f"{n_updates}.mp4")


class PPO_Agent(recording.Agent):
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def get_action(self, obs) -> t.Any:
        obs_prep, _ = self.preprocessor.process(np.asarray(obs), None)
        with torch.no_grad():
            return self.model.action(torch.FloatTensor(obs_prep), greedy=True).numpy()
