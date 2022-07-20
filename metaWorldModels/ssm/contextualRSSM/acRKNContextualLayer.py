import torch
from omegaconf import DictConfig, OmegaConf
from metaWorldModels.ssm.kalmanOps.update import Update
from metaWorldModels.ssm.kalmanOps.contextual_predict import AcPredict

nn = torch.nn


class AcRKNContextLayer(nn.Module):
    @staticmethod
    def get_default_config() -> OmegaConf:
        s = """
            num_basis: 15
            bandwidth: 3
            enc_net_hidden_units: 
                - 120
            dec_net_hidden_units: 
                - 120
            trans_net_hidden_units:
                - 120
            control_net_hidden_units:
                - 120
                - 120
                - 120
            process_noise_hidden_units:
                - 30
            trans_net_hidden_activation: "Tanh"
            control_net_hidden_activation: 'ReLU'
            process_noise_hidden_activation: 'ReLU'
            learn_trans_covar: False
            trans_covar: 0.1
            learn_initial_state_covar: True
            initial_state_covar: 10
            learning_rate: 7e-3
            enc_out_norm: 'post'
            clip_gradients: True
            never_invalid: False
            """

        return OmegaConf.create(s)

    def __init__(self, ltd, latent_obs_dim, action_dim, config=None, dtype=torch.float32,
                 use_cuda_if_available: bool = True):
        super().__init__()
        if config == None:
            self.c = self.get_default_config()
        else:
            self.c = config
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._ltd = ltd
        self._action_dim = action_dim
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._update = Update(self._lod, self.c)
        self._predict = AcPredict(self._ltd, self._lod, self._action_dim, self.c)

    def forward(self, latent_obs, obs_vars, action, latent_task, initial_mean, initial_cov, obs_valid=None):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param action: control signals
        :param latent_task: context generated usually from past m time steps for the B batches
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        ##TODO: multistep loss in trainig too
        """

        # if you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return

        prior_mean_list = []
        prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]

        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        for i in range(latent_obs.shape[1]):

            if i < latent_obs.shape[1]:
                cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None
                # print(obs_valid.shape)

                # update belief with updateLayer
                post_mean, post_cov = \
                    self._update(prior_mean, prior_cov, latent_obs[:, i], obs_vars[:, i], cur_obs_valid)

                post_mean_list.append(post_mean)
                [post_cov_list[i].append(post_cov[i]) for i in range(3)]

                # predict next belief state ahead in time
                next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i], latent_task)

                prior_mean_list.append(next_prior_mean)
                [prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]

                prior_mean = next_prior_mean
                prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = [torch.stack(x, 1) for x in prior_cov_list]

        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]

        return post_means, post_covs, prior_means, prior_covs
