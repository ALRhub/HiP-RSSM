import sys

sys.path.append('.')

import torch
import numpy as np
from utils.TimeDistributed import TimeDistributed
from metaWorldModels.ssm.ssmEncoderDecoder.Encoder import Encoder
from metaWorldModels.ssm.ssmEncoderDecoder.Decoder import SplitDiagGaussianDecoder, SimpleDecoder
from metaWorldModels.ssm.contextualRSSM.acRKNContextualLayer import AcRKNContextLayer
from omegaconf import OmegaConf
from typing import Tuple

optim = torch.optim
nn = torch.nn


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


def tanh2(x, min_y, max_y):
    scale_x = 1 / ((max_y - min_y) / 2)
    return (max_y - min_y) / 2 * (torch.tanh(x * scale_x) + 1.0) + min_y


class acrknContextualDecoder(nn.Module):

    def __init__(self, ltd: int, target_dim: int, lod: int, action_dim: int, config: OmegaConf = None,
                 use_cuda_if_available: bool = True):
        super(acrknContextualDecoder, self).__init__()

        """
               :param target_dim:
               :param lod:
               :param config:
               :param use_cuda_if_available:
               """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._ltd = ltd
        self._inp_dim = target_dim
        self._action_dim = action_dim
        self._lod = lod
        self._lsd = 2 * self._lod
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm
        # main model

        ###### ACRKN ENCODER LAYER OBJECTS DEFINED
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization, activation=self.c.variance_act)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        ###### ACRKN CELL OBJECT DEFINED
        self._rkn_layer = AcRKNContextLayer(ltd=self._ltd, latent_obs_dim=lod, action_dim=action_dim,
                                            config=self.c).to(self._device)

        ###### ACRKN DECODER OBJECT DEFINED
        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        # SimpleDecoder._build_hidden_layers = self._build_dec_hidden_layers
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(out_dim=target_dim, activation=self.c.variance_act),
                                    num_outputs=2).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

        ##### build (default) initial state

        if self.c.learn_initial_state_covar:
            init_state_covar = elup1_inv(self.c.initial_state_covar)
            self._init_state_covar_ul = \
                nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd), init_state_covar))
        else:
            self._init_state_covar_ul = self.c.initial_state_covar * torch.ones(1, self._lsd)

        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._init_state_covar_ul[:, :self._lod].to(self._device))
        self._icl = torch.nn.Parameter(self._init_state_covar_ul[:, self._lod:].to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self):
        layers = []
        last_hidden = self._inp_dim
        # hidden layers
        for hidden_dim in self.c.enc_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers_mean(self):
        layers = []
        if self.c.decoder_conditioning:
            last_hidden = self._lod * 2 + int(self._ltd / 2)
        else:
            last_hidden = self._lod * 2
        # hidden layers
        for hidden_dim in self.c.dec_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers_var(self):
        layers = []
        if self.c.decoder_conditioning:
            last_hidden = self._lod * 3 + int(self._ltd / 2)
        else:
            last_hidden = self._lod * 3
        # hidden layers
        for hidden_dim in self.c.dec_net_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.25))
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden


    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, latent_task: torch.Tensor,
                obs_valid_batch: torch.Tensor, multiStep=0, decode=True) -> Tuple[float, float]:
        """Single update step on a batch
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :param target_batch: batch of target sequences
        :param decode: whether to decode next_prior
        """
        conditional = False
        latent_task_mu = torch.unsqueeze(latent_task, 1).repeat(1, obs_batch.shape[1], 1) #only for conditional cases
        if conditional:
            w, w_var = self._enc(torch.cat([obs_batch, latent_task_mu[:, :, :self._ltd]], dim=-1))
        else:
            w, w_var = self._enc(obs_batch)
        post_mean, post_cov, prior_mean, prior_cov = self._rkn_layer(w, w_var, act_batch, latent_task,
                                                                     self._initial_mean,
                                                                     [self._icu, self._icl, self._ics], obs_valid_batch)
        if decode:
            if self.c.decoder_conditioning:
                out_mean, out_var = self._dec(
                    torch.cat([prior_mean, latent_task_mu[:, :, :int(self._ltd / 2)]], dim=-1),
                    torch.cat([torch.cat(prior_cov, dim=-1), latent_task_mu[:, :, int(self._ltd / 2):]], dim=-1))
            else:
                out_mean, out_var = self._dec(prior_mean, torch.cat(prior_cov, dim=-1))
                # out_mean = self._dec(prior_mean)

            return out_mean, out_var
        else:
            return prior_mean, prior_cov




# cell_conf = AcRKNCell.get_default_config()
# AcRKN = acrknContextGen(2,3,4,cell_conf)
# print(AcRKN)
# for name, param in AcRKN.named_parameters():
#     if param.requires_grad:
#         print(name)
