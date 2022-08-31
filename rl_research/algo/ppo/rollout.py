from collections import namedtuple
import warnings

from gym.spaces import Discrete
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


"""
      0    1    2     3    4    5    6    7    8    9    10   11
    @---------------@-----------------------------@--------------@
    | o0 - o1 - o2  | o0 - o1 - o2 - o3 - o4 - o5 | o0 - o1 ~ o2 | obs_prev
    | v0 - v1 - v2  | v0 - v1 - v2 - v3 - v4 - v5 | v0 - v1 ~ vn | value_pred
    | l0 - l1 - l2  | l0 - l1 - l2 - l3 - l4 - l5 | l0 - l1 ~    | log_pi
    | r0 - r1 - r2  | r0 - r1 - r2 - r3 - r4 - r5 | r0 - r1 ~    | reward
    | a0 - a1 - a2  | a0 - a1 - a2 - a3 - a4 - a5 | a0 - a1 ~    | action
    |    - d0 - d1  |    - d0 - d1 - d2 - d3 - d4 |    - d0 ~ d1 | done
    @---------------@-----------------------------@--------------@
    | o1 - o2 - o3  | o1 - o2 - o3 - o4 - o5 - o6 | o0 - o1 ~    | obs_cur

"""

class TorchStorage:
    """
        Simple storage for box-like observation spaces and
            either Discrete or 1-d action spaces.
    """

    _transferable = [
        "obs", "value_pred", "returns",
        "h_states", "log_pi", "rewards",
        "actions",
    ]
    _rollout_spec = [
        "obs", "value_pred", #"h_states",
        "log_pi", "rewards", "dones",
        "actions",
    ]

    #TODO: recurrent generator,
    def __init__(self, traj_len, n_actors, obs_shape, act_space, h_state_size):
        """
            Currently recurrent stuff is not supported!!
        """
        self._st = {
            "obs":        torch.zeros(traj_len, n_actors, *obs_shape),
            "h_states":   torch.zeros(traj_len, n_actors, h_state_size),
            "dones":      torch.zeros(traj_len + 1, n_actors, 1),
            "returns":    torch.zeros(traj_len + 1, n_actors, 1),
            "value_pred": torch.zeros(traj_len + 1, n_actors, 1),

            "log_pi":     torch.zeros(traj_len, n_actors, 1),
            "rewards":    torch.zeros(traj_len, n_actors, 1),
        }
        if isinstance(act_space, Discrete):
            self._st["actions"] = torch.zeros(traj_len, n_actors, 1).long()
        else:
            self._st["actions"] = torch.zeros(traj_len, n_actors,
                                              act_space.shape[0])

        self.traj_len = traj_len
        self.n_actors = n_actors
        self.i_step = 0

    def to(self, device):
        for name in self._transferable:
            self._st[name] = self._st[name].to(device)

    def save_rollout(self, rollout: dict):
        if set(rollout.keys()) != set(self._rollout_spec):
            warnings.warn("Incomplete rollout")
        for name in rollout:
            self._st[name][self.i_step].copy_(rollout[name])

        self.i_step = (self.i_step + 1) % self.traj_len

    def finalize_storage(self, value_pred, done):
        self._st["value_pred"][-1].copy_(value_pred)
        self._st["dones"][-1].copy_(done)

    def get_normalized_advantage(
        self,
        use_gae=False,
        discount_factor=.99,
        gae_lambda=.95
    ):
        m    = 1 - self._st["dones"]
        r    = self._st["rewards"]
        rets = self._st["returns"]
        val  = self._st["value_pred"]

        if use_gae:
            gae = 0
            for i in reversed(range(self.traj_len)):
                delta = r[i] + discount_factor * val[i + 1] * m[i + 1] - val[i]
                gae = delta + discount_factor * gae_lambda * m[i + 1] * gae
                # gae *= is_good[i + 1]
                rets[i] = gae + val[i]
        else:
            rets[-1] = val[-1]
            for i in reversed(range(self.traj_len)):
                rets[i] = rets[i + 1] * discount_factor * m[i + 1] + r[i]
                # * (1 - is_good[i + 1]) + good[i + 1] * (1 - m[i + 1]
        adv = rets[:-1] - val[:-1]
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def generate_fwd(
        self,
        adv,
        batch_size
    ):
        adv_target = None
        for idx in BatchSampler(
            SubsetRandomSampler(range(self.traj_len * self.n_actors)),
            batch_size,
            drop_last=True
        ):
            if adv is not None:
                adv_target = adv.view(-1, 1)[idx]

            yield (
                self._st["obs"].view(-1, *self._st["obs"].size()[2:])[idx],
                self._st["h_states"].view(-1, self._st["h_states"].size(-1))[idx],
                self._st["actions"].view(-1, self._st["actions"].size(-1))[idx],
                self._st["value_pred"][:-1].view(-1, 1)[idx],
                self._st["returns"][:-1].view(-1, 1)[idx],
                1 - self._st["dones"][:-1].view(-1, 1)[idx],
                self._st["log_pi"].view(-1, 1)[idx],
                adv_target
            )
