import torch
import torch.distributions as dist

def load_ckpt(path):
    with open(str(path)) as fp:
        return torch.load(fp)

class AC_Categorical:
    ## Actor-Critic Categorical Fully Connected
    ## discr -> (traj_len, n_actors, action_shape)

    def __init__(
        self,
        feature_extractor,
        policy_head,
        value_head,
        shared_extractor=True
    ):
        self.feature_extractor = feature_extractor
        self.policy_head = policy_head
        self.value_head = value_head
        self.shared_extractor = shared_extractor

        self.parameters = (
            [_ for _ in self.feature_extractor.parameters()] +
            [_ for _ in self.policy_head.parameters()] +
            [_ for _ in self.value_head.parameters()]
        )

    def policy_and_value(self, obs):
        features = self.feature_extractor(obs)
        log_pi = self.policy_head(features)

        if self.shared_extractor:
            value_pred = self.value_head(features)
        else:
            value_pred = self.value_head(obs)

        return log_pi, value_pred

    def action(self, obs, greedy=False):
        features = self.feature_extractor(obs)
        log_pi = self.policy_head(features)
        act, log_pi_chosen = self.select_action(log_pi, greedy)
        return act

    def value(self, obs):
        if self.shared_extractor:
            features = self.feature_extractor(obs)
            value_pred = self.value_head(features)
        else:
            value_pred = self.value_head(obs)

        return value_pred

    def select_action(self, log_pi, action=None, greedy=False):
        shape = (*log_pi.shape[:-1], 1)

        if greedy:
            m = torch.max(log_pi, dim=-1)
            log_pi_chosen = m[0]
            act_chosen = m[1]
            return act_chosen, log_pi_chosen
        elif action is None:
            pi = torch.nn.functional.softmax(log_pi, dim=-1)
            d = dist.Categorical(pi.view(-1, log_pi.shape[-1]))
            idx = d.sample().view(*shape)
        else:
            idx = action

        log_pi_chosen = torch.gather(log_pi, -1, idx)
        return idx.view(idx.shape[:-1]), log_pi_chosen.view(shape[:-1])

    def entropy(self, log_pi):
        d = dist.Categorical(log_pi.view(-1, log_pi.shape[-1]))
        return d.entropy().view(log_pi.shape[:-1])

    def save(self, path):
        torch.save(self, str(path))


class AC_ContNormal:
    ## cont -> (traj_len, n_actors, action_shape), sigma

    def __init__(
        self,
        action_size,
        feature_extractor,
        policy_head,
        value_head,
        shared_extractor=True
    ):
        self.feature_extractor = feature_extractor
        self.policy_head = policy_head
        self.value_head = value_head
        self.shared_extractor = shared_extractor
        self.action_size = action_size
        self.logvar = torch.nn.Parameter(torch.zeros(action_size))

        self.parameters = (
            [_ for _ in self.feature_extractor.parameters()] +
            [_ for _ in self.policy_head.parameters()] +
            [_ for _ in self.value_head.parameters()] +
            [self.logvar]
        )

    def policy_and_value(self, obs):
        features = self.feature_extractor(obs)
        mean = self.policy_head(features)

        if self.shared_extractor:
            value_pred = self.value_head(features)
        else:
            value_pred = self.value_head(obs)

        return mean, value_pred

    def action(self, obs, greedy=False):
        features = self.feature_extractor(obs)
        means = self.policy_head(features)
        act, _ = self.select_action(means, greedy=greedy)
        return act

    def value(self, obs):
        if self.shared_extractor:
            features = self.feature_extractor(obs)
            value_pred = self.value_head(features)
        else:
            value_pred = self.value_head(obs)

        return value_pred

    def covariance_matrix(self):
        return torch.diag(self.logvar.exp())

    def select_action(self, means, action=None, greedy=False):
        d = dist.Normal(means, self.logvar.exp())
        d = dist.Independent(d, 1)
        if greedy:
            action = means
        elif action is None:
            action = d.sample()

        log_pi = d.log_prob(action)
        #print(means - action)
        return action, log_pi

    def entropy(self, means):
        d = dist.Normal(means, self.logvar.exp())
        d = dist.Independent(d, 1)
        return d.entropy()

    def save(self, path):
        torch.save(self, str(path))
