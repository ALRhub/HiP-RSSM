from typing import Tuple

import numpy as np
import torch
from metaWorldModels.ssm.LSTMBaseline import LSTMBaseline
from torch.utils.data import TensorDataset, DataLoader
from utils.dataProcess import split_k_m, get_sliding_context_batch_mbrl, get_ctx_target_multistep, get_ctx_target_impute,\
    squeeze_sw_batch, diffToStateMultiStep, diffToState, diffToStateImpute

optim = torch.optim
nn = torch.nn


class Infer:

    def __init__(self, model: LSTMBaseline,  normalizer= None, config = None, run = None, log=True, use_cuda_if_available: bool = True):

        """
        :param model: nn module for acrkn
        :param use_cuda_if_available:  if to use gpu
        """
        assert run is not None, 'Enter a valid wandb run'
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._model = model
        self._normalizer = normalizer
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = log
        if self._log:
            self._run = run


    def predict(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor, imp: float = 0.0, k=32, test_gt_known=True,
                batch_size: int = -1, multiStep=0, tar="observations") -> Tuple[float, float]:
        '''
        :param obs:
        :param act:
        :param y_context:
        :param imp:
        :param test_gt_known:
        :param batch_size:
        :param multiStep:
        :return:
        '''
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        out_mean_list = []
        out_var_list = []
        cur_obs_list = []
        gt_list = []
        obs_valid_list = []
        self._context_size = k
        dataset = TensorDataset(obs, act, y_context)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs, act, target) in enumerate(loader):
            with torch.no_grad():
                # Assign data tensors to devices
                obs_batch = (obs).to(self._device)
                act_batch = act.to(self._device)
                target_batch = (target).to(self._device)

                # Split to context and targets
                if self._context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else:
                    k = self._context_size
                m = obs_batch.shape[1] - k
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, test_gt_known= test_gt_known,
                                          tar_imp=imp,
                                          random_seed=True)

                ### Unlike in learning during inference we don't have access to Y_target
                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)



                # Forward Pass
                out_mean, out_var = self._model(tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                # Diff To State

                if tar == "delta":
                    out_mean = \
                        torch.from_numpy(diffToStateImpute(out_mean, tar_obs_batch, tar_obs_valid_batch, self._normalizer,
                                             standardize=True)[0])
                    tar_tar_batch = \
                        torch.from_numpy(diffToState(tar_tar_batch, tar_obs_batch, self._normalizer, standardize=True)[0])

                out_mean_list.append(out_mean.cpu())
                out_var_list.append(out_var.cpu())
                gt_list.append(tar_tar_batch.cpu())  # if test_gt_known flag is False then we get list of Nones
                obs_valid_list.append(tar_obs_valid_batch.cpu())
                cur_obs_list.append(tar_obs_batch.cpu())
        return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list), torch.cat(obs_valid_list), torch.cat(cur_obs_list)

    def predict_multiStep(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor, k=32, test_gt_known=True,
                batch_size: int = -1, multiStep=0) -> Tuple[float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param y_context: the label information for the context sets
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        self._context_size = k
        out_mean_list = []
        out_var_list = []
        gt_list = []
        dataset = TensorDataset(obs, act, y_context)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs_batch, act_batch, targets_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)

                # Split to context and targets
                if self._context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else:
                    k = self._context_size
                m = obs_batch.shape[1] - k
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_multistep(obs_batch, act_batch, target_batch, k, num_context=None, test_gt_known=test_gt_known, tar_burn_in=5,
                                          random_seed=True)

                ### Unlike in learning during inference we don't have access to Y_target
                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)

                # Forward Pass
                out_mean, out_var = self._model(tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                out_mean_list.append(out_mean.cpu())
                out_var_list.append(out_var.cpu())
                gt_list.append(tar_tar_batch.cpu()) #if test_gt_known flag is False then we get list of Nones


        return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)

    def predict_mbrl(self, obs: torch.Tensor, act: torch.Tensor, y_context: torch.Tensor, k=32,
                batch_size: int = -1, multiStep=0, tar="observations") -> Tuple[float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param y_context: the label information for the context sets
        :param batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        :param multiStep: how many multiStep ahead predictions do you need. You can also do this by playing with obs_valid flag.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        self._context_size = k
        out_mean_list = []
        out_var_list = []
        gt_list = []
        dataset = TensorDataset(obs, act, y_context)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs_batch, act_batch, target_batch) in enumerate(loader):
            # Assign tensors to devices
            obs_batch = (obs_batch).to(self._device)
            act_batch = act_batch.to(self._device)
            target_batch = (target_batch).to(self._device)

            with torch.no_grad():
                # Split to context and targets
                if self._context_size is None:
                    k = int(obs_batch.shape[1] / 2)
                else:
                    k = self._context_size
                m = obs_batch.shape[1] - k
                tar_burn_in = 100

                # get sliding window(sw) batches based to k+steps+tar_burn_in episode length
                sw_obs, sw_act, sw_target = get_sliding_context_batch_mbrl(obs_batch, act_batch, target_batch, k , steps=multiStep,
                                                                          tar_burn_in=tar_burn_in)

                # Assign tensors to devices
                sw_obs = (sw_obs).to(self._device)
                sw_act = sw_act.to(self._device)
                sw_target = (sw_target).to(self._device)

                # Split into context and targets
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_multistep(sw_obs, sw_act, sw_target, k, num_context=None, tar_burn_in=tar_burn_in,
                                          random_seed=True)
                ### Unlike in learning during inference we don't have access to Y_target
                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)

                # Forward Pass
                out_mean, out_var = self._model(tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                # Diff To State
                if tar == "delta":
                    out_mean = \
                    diffToStateMultiStep(out_mean, tar_obs_batch, tar_obs_valid_batch, self._normalizer, standardize=True)[0]
                    tar_tar_batch = \
                        diffToState(tar_tar_batch, tar_obs_batch, self._normalizer, standardize=True)[0]

                # Squeeze To Original Episode From Hyper Episodes

                squeezed_mean, squeezed_var, squeezed_gt = squeeze_sw_batch(out_mean, out_var, tar_tar_batch,
                                                                               num_episodes=obs_batch.shape[0])

                #

                out_mean_list.append(squeezed_mean.cpu())
                out_var_list.append(squeezed_var.cpu())
                gt_list.append(squeezed_gt.cpu())

        return torch.cat(out_mean_list), torch.cat(out_var_list), torch.cat(gt_list)


