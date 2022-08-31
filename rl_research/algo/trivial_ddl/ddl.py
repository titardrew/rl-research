from dataclasses import dataclass
import typing as t
import time

import gym
import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler

import rl_research.recording as recording


def gershgorin_loss(matr: torch.Tensor, eps: float = 1e-8):
    diagonal = matr.diag()
    row_reduced = torch.sum(torch.abs(matr), dim=1)
    abs_diagonal = torch.abs(diagonal)

    # Summation trick to avoid summing over the diagonal
    R = row_reduced - abs_diagonal

    # Upper bound on lambda on real axis
    lambda_ub = diagonal + R + eps

    return torch.nn.functional.relu(lambda_ub).sum()


def is_eigen_stable(matr: np.ndarray):
    e, v = np.linalg.eig(matr)
    stable = True
    for i in range(e.shape[0]):
        r = np.real(e[i])
        if(r > 0.0):
            stable = False
    return stable


class Model:
    def __init__(self, input_obs_size: int, input_act_size: int,
                 hidden_size: int = 32, latent_size: int = 32):
        self.input_obs_size = input_obs_size
        self.input_act_size = input_act_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.obs_f = nn.Sequential(
            nn.Linear(input_obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.LayerNorm(latent_size),
        )

        self.act_g = nn.Sequential(
            nn.Linear(input_act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.LayerNorm(latent_size),
        )

        def normal_init(module):
            nn.init.normal_(module)
            return module

        self.param_A = normal_init(nn.Parameter(torch.zeros(latent_size, latent_size)))
        self.param_B = normal_init(nn.Parameter(torch.zeros(latent_size, latent_size)))

    def parameters(self):
        return list(self.obs_f.parameters()) + list(self.act_g.parameters()) + [self.param_A] + [self.param_B]

    @staticmethod
    def copy_spec_only(model):
        return Model(model.input_obs_size, model.input_act_size, model.hidden_size, model.latent_size)

    def dynamics_model(self, obs_latent: torch.Tensor, act_latent: torch.Tensor):
        obs_next_latent = F.linear(obs_latent, self.param_A) + F.linear(act_latent, self.param_B)
        return obs_next_latent

    def get_optimizer(self, lr: float = 1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-6)


@dataclass
class EpochStats:
    det_A: float
    det_B: float
    loss: float
    regression_loss: float
    gersh_A: float
    gersh_B: float
    is_stable_A: float
    is_stable_B: float
    time_seconds: float

    def __str__(self):
        return (
            f"(det(A)={self.det_A:.3f}, "
            f"det(B)={self.det_B:.3f}, "
            f"loss={self.loss:.7f}, "
            f"regression_loss={self.regression_loss:.7f}, "
            f"gersh(A)={self.gersh_A:.5f}, "
            f"gersh(B)={self.gersh_B:.5f}, "
            f"is_stable(A)={self.is_stable_A}, "
            f"is_stable(B)={self.is_stable_B}, "
            f"time={self.time_seconds:.1f}s)"
        )


def train_latent_one_epoch(
    model: Model,
    preprocessor,
    df_act_train: pd.DataFrame,
    df_obs_train: pd.DataFrame,
    df_obs_next_train: pd.DataFrame,
    df_act_val: t.Optional[pd.DataFrame] = None,
    df_obs_val: t.Optional[pd.DataFrame] = None,
    df_obs_next_val: t.Optional[pd.DataFrame] = None,
    lr: float = 1e-4,
    batch_size: int = 32,
    regression_weight: float = 1.0,
) -> t.Tuple[EpochStats, EpochStats]:

    def polyak_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def _compute_epoch(
            model: Model,
            preprocessor,
            df_act: pd.DataFrame,
            df_obs: pd.DataFrame,
            df_obs_next: pd.DataFrame,
            lr: float = 1e-4,
            batch_size: int = 32,
            regression_weight: float = 1.0,
            eval_mode=False,
    ) -> EpochStats:

        enable_grad = not eval_mode

        with torch.set_grad_enabled(enable_grad):
            if eval_mode:
                data_sampler_class = SequentialSampler
            else:
                data_sampler_class = SubsetRandomSampler
                optimizer = model.get_optimizer(lr)

            n_iters = 0
            total_loss = 0
            total_regression_loss = 0
            total_gersh_circle_loss_A = 0
            total_gersh_circle_loss_B = 0

            time_start = time.time()

            obs = df_obs.to_numpy()
            act = df_act.to_numpy()
            obs_next = df_obs_next.to_numpy()

            assert obs.shape[0] == act.shape[0] == obs_next.shape[0]
            num_data_points = obs.shape[0]

            fixed_model_ = Model.copy_spec_only(model)
            fixed_model_obs_f = fixed_model_.obs_f
            polyak_update(fixed_model_obs_f, model.obs_f, 1.0)

            num_updates = 0

            for idx in BatchSampler(data_sampler_class(range(num_data_points)), batch_size, drop_last=True):
                if num_updates % 500 == 0:
                    polyak_update(fixed_model_obs_f, model.obs_f, 1.0)

                batch_obs, batch_act, batch_obs_next = obs[idx], act[idx], obs_next[idx]

                inp_batch_obs = torch.FloatTensor(preprocessor.process(batch_obs, None)[0])
                out_batch_obs_next = torch.FloatTensor(preprocessor.process(batch_obs_next, None)[0])
                inp_batch_act = torch.FloatTensor(batch_act)

                obs_trans      = model.obs_f(inp_batch_obs)
                act_trans      = model.act_g(inp_batch_act)
                # obs_next_trans = model.obs_f(out_batch_obs_next).detach()
                obs_next_trans = fixed_model_obs_f(out_batch_obs_next).detach()

                obs_next_trans_pred = model.dynamics_model(obs_trans, act_trans)

                regression_loss = torch.norm(obs_next_trans - obs_next_trans_pred, dim=0).mean()
                gersh_circle_loss_A = gershgorin_loss(model.param_A)
                gersh_circle_loss_B = gershgorin_loss(model.param_B)
                loss = regression_loss * regression_weight + gersh_circle_loss_A + gersh_circle_loss_B

                if not eval_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    num_updates += 1

                # Recording stats
                total_loss += loss.detach().numpy().item()
                total_regression_loss += regression_loss.detach().numpy().item()
                total_gersh_circle_loss_A += gersh_circle_loss_A.detach().numpy().item()
                total_gersh_circle_loss_B += gersh_circle_loss_B.detach().numpy().item()
                n_iters += 1

            A, B = model.param_A.detach().numpy(), model.param_B.detach().numpy()

            time_finish = time.time()

        return EpochStats(
            det_A=np.linalg.det(A),
            det_B=np.linalg.det(B),
            loss=total_loss / n_iters,
            regression_loss=total_regression_loss / n_iters,
            gersh_A=total_gersh_circle_loss_A / n_iters,
            gersh_B=total_gersh_circle_loss_B / n_iters,
            is_stable_A=is_eigen_stable(A),
            is_stable_B=is_eigen_stable(B),
            time_seconds=time_finish - time_start)


    # train epoch
    train_epoch_stats = _compute_epoch(
        model, preprocessor,
        df_act_train, df_obs_train, df_obs_next_train,
        lr=lr, batch_size=batch_size, regression_weight=regression_weight
    )

    # evaluation epoch
    if df_act_val is not None and \
       df_obs_val is not None and \
       df_obs_next_val is not None:

        eval_epoch_stats = _compute_epoch(
            model, preprocessor,
            df_act_val, df_obs_val, df_obs_next_val,
            batch_size=batch_size, regression_weight=regression_weight,
            eval_mode=True
        )
    else:
        eval_epoch_stats = None

    return train_epoch_stats, eval_epoch_stats


class PolicyModel(nn.Module):
    def __init__(self, out_act_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_act_size),
        )

    def forward(self, obs_latent):
        return self.policy_net(obs_latent)

    def parameters(self):
        return self.policy_net.parameters()

    def get_optimizer(self, lr: float = 1e-4):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-6)


@dataclass
class PolicyEpochStats:
    loss: float
    time_seconds: float

    def __str__(self):
        return (
            f"(loss={self.loss:.7f}, "
            f"time={self.time_seconds:.2f}s)"
        )


def train_policy_one_epoch(
    model: Model,
    policy_model: PolicyModel,
    preprocessor,
    df_act_train: pd.DataFrame,
    df_obs_train: pd.DataFrame,
    df_obs_next_train: pd.DataFrame,
    df_act_val: t.Optional[pd.DataFrame] = None,
    df_obs_val: t.Optional[pd.DataFrame] = None,
    df_obs_next_val: t.Optional[pd.DataFrame] = None,
    lr: float = 1e-4,
    batch_size: int = 32,
) -> None:

    optimizer = policy_model.get_optimizer(lr)

    def _compute_epoch(
            model: Model,
            policy_model: PolicyModel,
            preprocessor,
            df_act: pd.DataFrame,
            df_obs: pd.DataFrame,
            df_obs_next: pd.DataFrame,
            lr: float = 1e-4,
            batch_size: int = 32,
            eval_mode=False,
    ) -> PolicyEpochStats:

        enable_grad = not eval_mode

        with torch.set_grad_enabled(enable_grad):
            if eval_mode:
                data_sampler_class = SequentialSampler
            else:
                optimizer = policy_model.get_optimizer(lr)
                data_sampler_class = SubsetRandomSampler

            n_iters = 0
            total_loss = 0

            time_start = time.time()

            for idx in BatchSampler(data_sampler_class(range(df_act.shape[0])), batch_size, drop_last=True):
                batch_obs, batch_act, batch_obs_next = df_obs.iloc[idx], df_act.iloc[idx], df_obs_next.iloc[idx]

                inp_batch_obs = torch.FloatTensor(preprocessor.process(batch_obs.to_numpy(), None)[0])
                out_batch_act = torch.FloatTensor(batch_act.to_numpy())

                obs_latent = model.obs_f(inp_batch_obs)
                pred_acts = policy_model(obs_latent)

                regression_loss = torch.norm(pred_acts - out_batch_act, dim=0).mean()
                loss = regression_loss

                if not eval_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Recording stats
                total_loss += loss.detach().numpy().item()
                n_iters += 1

            time_finish = time.time()

            return PolicyEpochStats(
                loss=total_loss / n_iters,
                time_seconds=time_finish - time_start,
            )

    # train epoch
    train_epoch_stats = _compute_epoch(
        model, policy_model, preprocessor,
        df_act_train, df_obs_train, df_obs_next_train,
        lr=lr, batch_size=batch_size)

    # train epoch
    if df_act_val is not None and \
       df_obs_val is not None and \
       df_obs_next_val is not None:

        eval_epoch_stats = _compute_epoch(
            model, policy_model, preprocessor,
            df_act_val, df_obs_val, df_obs_next_val,
            batch_size=batch_size, eval_mode=True)
    else:
        eval_epoch_stats = None

    return train_epoch_stats, eval_epoch_stats


class Agent(recording.Agent):
    def __init__(self, model: Model, policy_model: PolicyModel, preprocessor):
        self.model = model
        self.policy_model = policy_model
        self.preprocessor = preprocessor

    def get_action(self, obs) -> t.Any:
        obs_prep, _ = self.preprocessor.process(np.asarray(obs), None)
        with torch.no_grad():
            inp_obs = torch.FloatTensor(obs_prep)
            obs_latent = self.model.obs_f(inp_obs)
            pred_acts = self.policy_model(obs_latent)
            act = pred_acts.numpy()
        return act
