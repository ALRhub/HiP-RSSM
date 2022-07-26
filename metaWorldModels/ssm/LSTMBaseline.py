import torch
import numpy as np
from utils.TimeDistributed import TimeDistributed
import time as t
from metaWorldModels.ssm.ssmEncoderDecoder.Encoder import EncoderSimple
from metaWorldModels.ssm.ssmEncoderDecoder.Decoder import SimpleDecoder
from typing import Tuple
optim = torch.optim
nn = torch.nn


class LSTMBaseline(nn.Module):

    def __init__(self, obs_dim:int, act_dim:int, target_dim: int, lod: int, cell_config = None, use_cuda_if_available: bool = True):
        """
        TODO: Gradient Clipping?
        :param target_dim:
        :param lod:
        :param cell_config:
        :param use_cuda_if_available:
        """
        super(LSTMBaseline, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._obs_dim = obs_dim
        self._act_dim = act_dim

        self._lod = lod
        self._lsd = 2 * self._lod
        if cell_config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = cell_config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm

        # main model
        # Its not ugly, its pythonic :)
        EncoderSimple._build_hidden_layers = self._build_obs_hidden_layers
        obs_enc = EncoderSimple(lod, output_normalization=self._enc_out_normalization)
        EncoderSimple._build_hidden_layers = self._build_act_hidden_layers
        act_enc = EncoderSimple(lod, output_normalization=self._enc_out_normalization)
        EncoderSimple._build_hidden_layers = self._build_enc_hidden_layers
        enc = EncoderSimple(self._lsd, output_normalization=self._enc_out_normalization)
        self._obs_enc = TimeDistributed(obs_enc, num_outputs=1).to(self._device)
        self._act_enc = TimeDistributed(act_enc, num_outputs=1).to(self._device)
        self._enc = TimeDistributed(enc, num_outputs=1).to(self._device)

        if self.c.gru:
            self._lstm_layer = nn.GRU(input_size= 2 * lod, hidden_size=5 * lod, batch_first=True).to(self._device)
        else:
            self._lstm_layer = nn.LSTM(input_size=2 * lod, hidden_size=5 * lod, batch_first=True).to(self._device)

        SimpleDecoder._build_hidden_layers = self._build_dec_hidden_layers
        self._dec = TimeDistributed(SimpleDecoder(out_dim=target_dim), num_outputs=2).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_obs_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_act_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError


    def forward(self, obs_batch: torch.Tensor, act_batch: torch.Tensor, obs_valid_batch: torch.Tensor) -> Tuple[float, float]:
        """Forward Pass oF RKN
        :param obs_batch: batch of observation sequences
        :param act_batch: batch of action sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :return: mean and variance
        """

        # here masked values are set to zero. You can also put an unrealistic value like a negative number.
        obs_masked_batch = obs_batch * obs_valid_batch
        w_obs = self._obs_enc(obs_masked_batch)
        w_obs = w_obs
        act_obs = self._act_enc(act_batch)
        input_batch = torch.cat([w_obs,act_obs], dim=-1)
        w = self._enc(input_batch)
        z, y = self._lstm_layer(w)

        out_mean, out_var = self._dec(z)

        if self.c.get_latent:
            return out_mean, out_var, prior_mean, prior_cov
        else:
            return out_mean, out_var

