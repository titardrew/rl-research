from torch import nn

from normal import *
from networks import *

torch.set_default_dtype(torch.float64)


class E2C_N(nn.Module):
    def __init__(self, obs_dim, z_dim, u_dim, n_steps, env = 'planar'):
        super(E2C_N, self).__init__()
        enc, dec, trans = load_config(env)

        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.u_dim = u_dim
        self.n_steps = n_steps

        self.encoder = enc(obs_dim=obs_dim, z_dim=z_dim)
        # self.encoder.apply(init_weights)
        self.decoder = dec(z_dim=z_dim, obs_dim=obs_dim)
        # self.decoder.apply(init_weights)
        self.trans = trans(z_dim=z_dim, u_dim=u_dim)
        # self.trans.apply(init_weights)

    def encode(self, x):
        """
        :param x:
        :return: mean and log variance of q(z | x)
        """
        return self.encoder(x)

    def decode(self, z):
        """
        :param z:
        :return: bernoulli distribution p(x | z)
        """
        return self.decoder(z)

    def transition(self, z_bar, q_z, u):
        """
        :param z_bar:
        :param q_z:
        :param u:
        :return: samples z_hat_next and Q(z_hat_next)
        """
        return self.trans(z_bar, q_z, u)

    def reparam(self, mean, logvar):
        sigma = (logvar / 2).exp()
        epsilon = torch.randn_like(sigma)
        return mean + torch.mul(epsilon, sigma)

    def forward(self, X, U):
        assert self.n_steps == len(U) == len(X) - 1

        X_next = X[1:]
        x0 = X[0]

        mu0, logvar0 = self.encode(x0)
        z0 = self.reparam(mu0, logvar0)
        q_z0 = NormalDistribution(mu0, logvar0)
        x0_recon = self.decode(z0)


        X_next_pred = []
        Q_z_next = []
        Q_z_next_pred = []

        q_z = q_z0
        z = z0

        Z_next_Q_z_next_pred, g_A, g_B = self.trans.forward_n(z, q_z, U, self.n_steps)

        for i_step in range(self.n_steps):
            z_next, q_z_next_pred = Z_next_Q_z_next_pred[i_step]

            x_next_pred = self.decode(z_next)

            mu_next, logvar_next = self.encode(X[i_step + 1])
            q_z_next = NormalDistribution(mean=mu_next, logvar=logvar_next)

            X_next_pred.append(x_next_pred)
            Q_z_next.append(q_z_next)
            Q_z_next_pred.append(q_z_next_pred)

            z = z_next
            q_z = q_z_next_pred


        return x0, x0_recon, q_z0, X_next, X_next_pred, Q_z_next, Q_z_next_pred, g_A, g_B

    def predict(self, x, u):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        q_z = NormalDistribution(mu, logvar)

        z_next, q_z_next_pred = self.transition(z, q_z, u)

        x_next_pred = self.decode(z_next)
        return x_next_pred
