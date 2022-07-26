import torch
from utils.ConfigDict import ConfigDict
from omegaconf import DictConfig, OmegaConf

nn = torch.nn

import torch
from typing import Tuple

nn = torch.nn

def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)

class SetEncoder(nn.Module):

    def __init__(self, input_dim:int, lod: int,  config: ConfigDict = None, use_cuda_if_available: bool = True):
        """The set encoder that calculates the posterior over latent context.
        :param input_dim: input dimension of each set element
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param config: dict of config
        :param use_cuda_if_available: if gpu training set to True
        """
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._input_dim = input_dim
        self._lod = lod
        self._num_hidden_list = self.c.encoder_hidden_units
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                             "not found by the optimizer"
        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._softplus = nn.Softplus()

        self._output_normalization = self.c.enc_out_norm
        self._aggregator = self.c.aggregator
        self._activation = self.c.variance_act


    def aggregate(self, obs_mean, obs_cov, type='BA'):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.
        Parameters
        ----------
        obs_mean : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        obs_cov : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        if type=='BA':
            # create intial state
            initial_mean, initial_cov = 0, 1
            initial_mean = torch.ones(self._lod, dtype=torch.float32, device=self._device) * initial_mean
            initial_cov = torch.ones(self._lod, dtype=torch.float32, device=self._device) * initial_cov

            # add task and states dimensions
            initial_mean = initial_mean[None, None, :]
            initial_cov = initial_cov[None, None, :]


            v = obs_mean - initial_mean
            cov_w_inv = 1 / obs_cov
            cov_z_new = 1 / (1 / initial_cov + torch.sum(cov_w_inv, dim=1))
            mu_z_new = initial_mean + cov_z_new * torch.sum(cov_w_inv * v, dim=1)
            return torch.squeeze(mu_z_new), torch.squeeze(cov_z_new)

        if type=='MA':
            mu_z_new = torch.mean(obs_mean, dim=1)
            return mu_z_new


    def _flatten(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)
        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        self.batch_size, self.num_points, self.x_dim = x.size()
        self.y_dim = y.size()[-1]
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(self.batch_size * self.num_points, self.x_dim)
        y_flat = y.contiguous().view(self.batch_size * self.num_points, self.y_dim)
        return x_flat, y_flat


    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        layers = []
        last_hidden = self._input_dim
        # hidden layer 1
        for hidden_dim in self._num_hidden_list:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x_flat, y_flat = self._flatten(x,y)

        h = torch.cat((x_flat,y_flat),dim=-1)


        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean_flat = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        log_var = self._log_var_layer(h)
        if self._activation == 'softplus':
            var_flat = torch.add(self._softplus(log_var), 0.0001)
        else:
            var_flat = elup1(log_var)

        # revert to batches
        # Reshape tensors into batches
        mean = mean_flat.view(self.batch_size, self.num_points, self._lod)
        var = var_flat.view(self.batch_size, self.num_points, self._lod)
        # Aggregate representations r_i into a single representation r
        mu_z, cov_z = self.aggregate(mean, var, type=self._aggregator)
        # Return parameters of distribution
        return mu_z, cov_z