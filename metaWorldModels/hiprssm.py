import sys

sys.path.append('.')
import numpy as np
import torch
from omegaconf import OmegaConf

from metaWorldModels.setEncoders.setFunctionContext import SetEncoder
from metaWorldModels.ssm.contextualRSSM.contextualRKN import acrknContextualDecoder

optim = torch.optim
nn = torch.nn

class HipRSSM(nn.Module):
    @staticmethod
    def get_default_config() -> OmegaConf:
        config = """
        clip_gradients: True
        latent_obs_dim: 30
        agg_dim: 60
        """
        return OmegaConf.create(config)

    '''
    Note that the aggregator is considered as an integral part of the encoder
    '''
    def __init__(self, obs_dim, action_dim, target_dim, config=None, dtype=torch.float32,
                 use_cuda_if_available: bool = True):
        '''
        encoder: context encoder and aggregator
        decoder: target decoder
        dec_type: 'acrkn' or 'ffnn'
        latent_variance: If True, decode the task level variance as well
        '''
        super(HipRSSM, self).__init__()
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._ltd = self.c.hiprssm.task_dim
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._target_dim = target_dim
        # model architecture
        self._encoder = SetEncoder(input_dim=self._obs_dim+self._action_dim+self._target_dim,
            lod=self.c.hiprssm.task_dim, config=self.c.set_encoder).to(self._device)
        self._decoder = acrknContextualDecoder(ltd=self.c.hiprssm.task_dim, target_dim=self._target_dim, action_dim=self._action_dim,
                                         lod=self.c.hiprssm.latent_obs_dim, config=self.c.ssm_decoder).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches


    def forward(self, context_inp, context_out, target_inp, multiStep=0):
        ##### Encode context to latent space
        mu_z, cov_z = self._encoder(context_inp, context_out)

        ###### Sample from the distribution if we have a VAE style sampling procedure with reparameterization
        latent_task = torch.cat((mu_z,cov_z),dim=-1)
        target_obs, target_act, target_obs_valid = target_inp

        ##### Conditioned on global latent task represenation make predictions on targets
        mu_x, cov_x = self._decoder(target_obs,
                                              target_act, latent_task,
                                              target_obs_valid, multiStep)


        return mu_x, cov_x, mu_z, cov_z










