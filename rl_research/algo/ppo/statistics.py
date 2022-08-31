import time
import warnings

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Timer:
    _start = 0

    def start(self):
        self._start = time.perf_counter()

    def get_elapsed_fmt(self):
        return self.fmt(self.get_elapsed_seconds())

    def get_elapsed_seconds(self):
        return time.perf_counter() - self._start

    @staticmethod
    def fmt(t):
        m_secs = t * 1000
        mc_secs = t * 1000000
        if t > 0.5:
            d = t // 86400
            h = t // 3600
            m = (t % 3600) // 60
            s = (t % 60)
            if d > 0:
                return "%dd. %dh. %dm. %.1fs." % (d, h, m, s)
            elif h > 0:
                return "%dh. %dm. %.1fs." % (h, m, s)
            elif m > 0:
                return "%dm. %.1fs." % (m, s)
            else:
                return "%.3fs." % s
        elif m_secs > 0.5:
            return "%.3fms." % m_secs
        else:
            return "%.3fmcs." % mc_secs


def safe_reduce(func, arr: np.ndarray, err_msg):
    if len(arr) > 0:
        return func(arr)
    else:
        warnings.warn(err_msg)
        return 0


def safe_min(arr: np.ndarray, err_msg="safe_min got an empty array"):
    return safe_reduce(np.min, arr, err_msg)


def safe_max(arr: np.ndarray, err_msg="safe_max got an empty array"):
    return safe_reduce(np.max, arr, err_msg)


def safe_mean(arr: np.ndarray, err_msg="safe_mean got an empty array"):
    return safe_reduce(np.mean, arr, err_msg)


def safe_std(arr: np.ndarray, err_msg="safe_std got an empty array"):
    return safe_reduce(np.std, arr, err_msg)


class Statistics:
    #TODO(andrii): Refactor this crap!!!
    def __init__(self, n_workers, log_stdout=True, tb_run_experiment=None):
        self.n_workers = n_workers
        self.epi_rewards_buffer = [0 for _ in range(n_workers)]
        self.epi_rewards  = []
        self.step_rewards = []
        self.mean_actions = []

        self.epi_lens_buffer = [0 for _ in range(n_workers)]
        self.epi_lens = []

        self.ppo_loss        = []
        self.ppo_value_loss  = []
        self.ppo_policy_loss = []
        self.ppo_entropy     = []
        self.ppo_clip_frac   = []
        self.ppo_approx_kl   = []
        self.ppo_returns     = []
        self.ppo_advantages  = []
        self.ppo_log_pi      = []
        self.ppo_value_pred  = []
        self.ppo_logstd      = []
        self.ppo_act_diff    = []

        self.total_steps               = 0
        self.total_episodes            = 0
        self.total_updates             = 0
        self.steps_since_last_reduce   = 0
        self.updates_since_last_reduce = 0
        self.time_per_update           = 0

        self.stats = {}

        self.tb_writer = None
        if tb_run_experiment:
            self.tb_writer = SummaryWriter(tb_run_experiment)

        self.log_stdout = log_stdout

        self.timer = Timer()
        self.timer.start()

    def add_env_data(self, data):
        rewards = data["reward"]
        dones = data["done"]
        actions = data["actions"]
        self.time_per_env = data["time_per_env"] / self.n_workers
        self.total_steps += self.n_workers

        for i in range(self.n_workers):
            self.epi_rewards_buffer[i] += rewards[i]
            self.step_rewards.append(rewards[i]) 
            self.mean_actions.append(actions[i].mean())

            self.epi_lens_buffer[i] += 1
            self.steps_since_last_reduce += 1

            if dones[i]:
                self.epi_rewards.append(self.epi_rewards_buffer[i])
                self.epi_rewards_buffer[i] = 0

                self.epi_lens.append(self.epi_lens_buffer[i])
                self.epi_lens_buffer[i] = 0

                self.total_episodes += 1

    def add_agent_data(self, data):
        ppo_loss        = data["loss"]
        ppo_value_loss  = data["value_loss"]
        ppo_policy_loss = data["policy_loss"]
        ppo_entropy     = data["entropy"]
        ppo_clip_frac   = data["clip_frac"]
        ppo_approx_kl   = data["approx_kl"]
        ppo_log_pi      = data["log_pi"]
        ppo_value_pred  = data["value_pred"]
        ppo_advantages  = data["advantages"]
        ppo_returns     = data["returns"]
        ppo_logstd      = data["logstd"]
        ppo_act_diff    = data["act_diff"]

        self.ppo_loss.append(ppo_loss)
        self.ppo_value_loss.append(ppo_value_loss)
        self.ppo_policy_loss.append(ppo_policy_loss)
        self.ppo_entropy.append(ppo_entropy)
        self.ppo_clip_frac.append(ppo_clip_frac)
        self.ppo_approx_kl.append(ppo_approx_kl)
        self.ppo_log_pi.append(ppo_log_pi)
        self.ppo_value_pred.append(ppo_value_pred)
        self.ppo_returns.append(ppo_returns)
        self.ppo_advantages.append(ppo_advantages)
        self.ppo_logstd.append(ppo_logstd)
        self.ppo_act_diff.append(ppo_act_diff)

        self.updates_since_last_reduce += 1
        self.total_updates += 1

    def reduce(self, time_elapsed):
        time_since_last_reduce = self.timer.get_elapsed_seconds()
        stats = self.stats
        stats["n_steps/total"] = self.total_steps
        stats["n_steps/per_update"] = (
            self.steps_since_last_reduce
                / self.updates_since_last_reduce
        )
        stats["n_episodes/per_update"] = (
            (self.total_episodes - stats.get("n_episodes/total", 0))
                / self.updates_since_last_reduce
        )
        stats["n_episodes/total"] = self.total_episodes
        stats["n_updates"] = self.total_updates

        err_msg = ("Cannot reduce statistics. No full episodes "
                   "in trajectory storage. You may want to "
                   "increase traj_len, reduce n_workers or "
                   "validate rewards separately.")
        stats["reward/epi_min"]  = safe_min(self.epi_rewards, err_msg)
        stats["reward/epi_mean"] = safe_mean(self.epi_rewards, err_msg)
        stats["reward/epi_max"]  = safe_max(self.epi_rewards, err_msg)
        stats["reward/epi_std"]  = safe_std(self.epi_rewards, err_msg)

        stats["reward/step_min"]  = np.min(self.step_rewards)
        stats["reward/step_mean"] = np.mean(self.step_rewards)
        stats["reward/step_max"]  = np.max(self.step_rewards)
        stats["reward/step_std"]  = np.std(self.step_rewards)

        stats["len/min"]  = safe_min(self.epi_lens, err_msg)
        stats["len/mean"] = safe_mean(self.epi_lens, err_msg)
        stats["len/max"]  = safe_max(self.epi_lens, err_msg)
        stats["len/std"]  = safe_std(self.epi_lens, err_msg)

        stats["ppo/loss"]        = np.mean(self.ppo_loss)
        stats["ppo/value_loss"]  = np.mean(self.ppo_value_loss)
        stats["ppo/policy_loss"] = np.mean(self.ppo_policy_loss)
        stats["ppo/entropy"]     = np.mean(self.ppo_entropy)
        stats["ppo/clip_frac"]   = np.mean(self.ppo_clip_frac)
        stats["ppo/approx_kl"]   = np.mean(self.ppo_approx_kl)
        stats["ppo/value_pred"]  = np.mean(self.ppo_value_pred)
        stats["ppo/returns"]     = np.mean(self.ppo_returns)
        stats["ppo/advantages"]  = np.mean(self.ppo_advantages)
        stats["ppo/log_pi"]      = np.mean(self.ppo_log_pi)
        stats["ppo/logstd"]      = np.mean(self.ppo_logstd)
        stats["ppo/act_diff"]    = np.mean(self.ppo_act_diff)
        stats["ppo/actions"]     = np.mean(self.mean_actions)

        stats["time/elapsed"]      = time_elapsed
        stats["time/per_update"]   = time_since_last_reduce / self.total_updates
        stats["time/per_env_step"] = self.time_per_env / self.total_steps

        self.steps_since_last_reduce = 0
        self.updates_since_last_reduce = 0
        self.time_per_update = 0
        # self.epi_rewards_buffer = [0 for _ in range(self.n_workers)]
        self.epi_rewards = []
        self.step_rewards = []
        # self.epi_lens_buffer = [0 for _ in range(self.n_workers)]
        self.epi_lens = []

        self.timer.start()

    def dump(self):
        if self.tb_writer:
            for k, v in self.stats.items():
                self.tb_writer.add_scalar(k, v, self.total_steps)

        if self.log_stdout:
            max_k_len = max(map(len, self.stats.keys()))
            lens = []
            for k, v in self.stats.items():
                if k == "time/elapsed":
                    string = f"%-{max_k_len}s: %s"
                    string = string % (k, Timer.fmt(v))
                    print(string)
                elif "ppo/" in k:
                    fl_string = ("%.6f" % v).rstrip("0").rstrip(".")
                    string = f"%-{max_k_len}s: " + fl_string
                    string = string % (k)
                    print(string)
                else:
                    fl_string = ("%.3f" % v).rstrip("0").rstrip(".")
                    string = f"%-{max_k_len}s: " + fl_string
                    string = string % (k)
                    print(string)
                lens.append(len(string))
            print("-"*max(lens))

    def dump_video(self, vids, tag, fps=5, size=None):
        # import pdb; pdb.set_trace()
        if self.tb_writer:
            dtype = vids[0][0].dtype
            if dtype != "uint8":
                warnings.warn(f"Frame dtype has to be uint8, not {dtype}. "
                               "It will be min-max normalized and casted.")
                def norm_uint8(x):
                    x = np.array(x)
                    x = x - x.min()
                    x /= x.max()
                    x *= 255
                    return torch.tensor(x).permute(0, 3, 1, 2).unsqueeze(0) # T C H W

                norm_fn = norm_uint8
            else:
                norm_fn = lambda x: torch.tensor(x).permute(0, 3, 1, 2).unsqueeze(0)
            if size:
                for n_vid in range(len(vids)):
                    for n_fr in range(len(vids[n_vid])):
                        rec = vids[n_vid][n_fr]
                        if isinstance(size, int):
                            w = int(size * rec.shape[1] / rec.shape[0])
                            h = size
                        elif isinstance(size, tuple) or isinstance(size, list):
                            w, h = size
                        vids[n_vid][n_fr] = cv2.resize(rec, (w, h))
            for vid in vids:
                self.tb_writer.add_video(
                    tag,
                    norm_fn(vid),
                    self.total_steps,
                    fps=fps
                )
