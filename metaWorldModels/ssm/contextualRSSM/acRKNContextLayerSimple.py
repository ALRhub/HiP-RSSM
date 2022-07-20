import torch
import numpy as np
from dynamics_models.rkn_cell.kalman_ops.update_simple import Update
from utils.ConfigDict import ConfigDict
#from rkn_cell.kalman_ops.ac_predict import AcPredict
from dynamics_models.rkn_cell.kalman_ops.contextual_predict_simple import AcPredict
nn = torch.nn
import tsensor


class AcRKNContextLayer(nn.Module):

    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[],
            control_net_hidden_units=[15],
            process_noise_hidden_units=[15],
            trans_net_hidden_activation="Tanh",
            control_net_hidden_activation='ReLU',
            process_noise_hidden_activation='ReLU',
            learn_trans_covar=True,
            context_flag_coeff=True,
            context_flag_control=True,
            context_flag_noise=True,
            hyper_transition_matrix=True,
            multi_gaussian_l_transform = True,
            trans_covar=1,
            learn_initial_state_covar=True,
            initial_state_covar=1,
            learning_rate=7e-3,
            enc_out_norm='post',
            clip_gradients=True,
            never_invalid=False
        )
        config.finalize_adding()
        return config

    def __init__(self, ltd, latent_obs_dim, act_dim, config: ConfigDict = None, dtype=torch.float32):
        super().__init__()
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._ltd = ltd
        self._lad = act_dim
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._update = Update(self._lod, self.c)
        self._predict = AcPredict(self._ltd, self._lod, self._lad, self.c)
        #self._predict = AcPredict(self._lod, self._lad, self.c)

    def forward(self, latent_obs, obs_vars, action, latent_task, initial_mean, initial_cov, obs_valid=None, multiStep=0):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param action: control signals
        :param latent_task: context generated usually from past m time steps for the B batches
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        :param multiStep: applies only for inference currently:
        ##TODO: multistep loss in trainig too
        """

        # tif you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return

        prior_mean_list = []
        prior_cov_list = []

        post_mean_list = []
        post_cov_list = []


        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        for i in range(latent_obs.shape[1]):

            if i < latent_obs.shape[1]:
                cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
                #print(obs_valid.shape)

                # update belief with updateLayer
                post_mean, post_cov = \
                    self._update(prior_mean, prior_cov, latent_obs[:, i], obs_vars[:, i], cur_obs_valid)


                post_mean_list.append(post_mean)
                post_cov_list.append(post_cov)

                # predict next belief state ahead in time
                next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i],latent_task)

                #next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i])

                prior_mean_list.append(next_prior_mean)
                prior_cov_list.append(next_prior_cov)

                prior_mean = next_prior_mean
                prior_cov = next_prior_cov

            elif multiStep >0:
                cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
                # print(obs_valid.shape)

                # update belief by copying the previous prior

                post_mean, post_cov = prior_mean, prior_cov

                post_mean_list.append(post_mean)
                post_cov_list.append(post_cov)
                import numpy as np

                # predict next belief state ahead in time
                next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i], latent_task)

                # next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i])

                prior_mean_list.append(next_prior_mean)
                prior_cov_list.append(next_prior_cov)

                prior_mean = next_prior_mean
                prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = torch.stack(prior_cov_list, 1)

        post_means = torch.stack(post_mean_list, 1)
        post_covs = torch.stack(post_cov_list, 1)

        return post_means, post_covs, prior_means, prior_covs
