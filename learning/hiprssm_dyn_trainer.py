import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

from metaWorldModels.hiprssm import HipRSSM
from utils.dataProcess import split_k_m, get_ctx_target_impute
from utils.Losses import mse, gaussian_nll
from utils.PositionEmbedding import PositionEmbedding as pe
from utils import ConfigDict
from utils.plotTrajectory import plotImputation
from utils.latentVis import plot_clustering

optim = torch.optim
nn = torch.nn


class Learn:

    def __init__(self, model: HipRSSM, loss: str, imp: float = 0.0, config: ConfigDict = None, run = None, log=True, use_cuda_if_available: bool = True):
        """
        :param model: nn module for np_dynamics
        :param loss: type of loss to train on 'nll' or 'mse'
        :param imp: how much to impute
        :param use_cuda_if_available: if gpu training set to True
        """
        assert run is not None, 'pass a valid wandb run'
        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._loss = loss
        self._imp = imp
        self._model = model
        self._pe = pe(self._device)
        self._exp_name = run.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config

        self._learning_rate = self.c.learn.lr

        self._save_path = os.getcwd() + '/experiments/saved_models/' + run.name + '.ckpt'
        self._cluster_vis = self.c.learn.latent_vis

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._log = bool(log)
        self.save_model = self.c.learn.save_model
        if self._log:
            self._run = run

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_targets: np.ndarray, train_task_idx: np.ndarray,
                   batch_size: int)  -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param train_task_idx: task ids per episode
        :param batch_size: batch size for each gradient update
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_targets, train_task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        avg_loss = avg_metric_nll = avg_metric_mse = 0
        t0 = t.time()
        b = list(loader)[0]
        z_vis_list = []
        task_id_list = []

        for batch_idx, (obs, act, targets, task_id) in enumerate(loader):
            # Assign tensors to device
            obs_batch = (obs).to(self._device)
            act_batch = act.to(self._device)
            target_batch = (targets).to(self._device)
            task_id = (task_id).to(self._device)

            # Split to context and targets
            k = int(obs_batch.shape[1] / 2)
            m = obs_batch.shape[1] - k
            ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,
                                      random_seed=True)
            if batch_idx == 0:
                print("Fraction of Valid Target Observations:",
                      np.count_nonzero(tar_obs_valid_batch) / np.prod(tar_obs_valid_batch.shape))

            ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0], ctx_obs_batch.shape[1],1)
            ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)


            tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)
            context_Y = ctx_target_batch
            target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

            context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)


            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            out_mean, out_var, mu_z, cov_z = self._model(context_X, context_Y, target_X)

            ## Calculate Loss
            if self._loss == 'nll':
                loss = gaussian_nll(tar_tar_batch, out_mean, out_var)
            else:
                loss = mse(tar_tar_batch, out_mean)

            # Backward Pass
            loss.backward()

            # Clip Gradients
            if self.c.hiprssm.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            with torch.no_grad():  #
                metric_nll = gaussian_nll(tar_tar_batch, out_mean, out_var)
                metric_mse = mse(tar_tar_batch, out_mean)

                z_vis_list.append(mu_z.detach().cpu().numpy())
                task_id_list.append(task_id.detach().cpu().numpy())
            avg_loss += loss.detach().cpu().numpy()
            avg_metric_nll += metric_nll.detach().cpu().numpy()
            avg_metric_mse += metric_mse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an epoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        with torch.no_grad():
            self._tr_sample_gt = tar_tar_batch.detach().cpu().numpy()
            self._tr_sample_valid = tar_obs_valid_batch.detach().cpu().numpy()
            self._tr_sample_pred_mu = out_mean.detach().cpu().numpy()
            self._tr_sample_pred_var = out_var.detach().cpu().numpy()

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))

        #z_vis = 0
        z_vis = np.concatenate(z_vis_list, axis=0)
        task_labels = np.concatenate(task_id_list, axis=0)
        return avg_loss, avg_metric_nll, avg_metric_rmse, z_vis, task_labels, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, targets: np.ndarray, task_idx: np.ndarray,
             batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evaluate on
        :param targets: targets to evaluate on
        :param task_idx: task index
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        dataset = TensorDataset(obs, act, targets, task_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        avg_loss = avg_metric_nll = avg_metric_mse = 0.0
        avg_metric = 0.0
        z_vis_list = []
        task_id_list = []

        for batch_idx, (obs_batch, act_batch, targets_batch, task_id) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                target_batch = (targets_batch).to(self._device)

                # Split to context and targets
                k = int(obs_batch.shape[1] / 2)
                m = obs_batch.shape[1] - k
                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs_batch, act_batch, target_batch, k, num_context=None, tar_imp=self._imp,
                                          random_seed=True)
                ctx_obs_valid_batch = torch.ones(ctx_obs_batch.shape[0],ctx_obs_batch.shape[1],1)
                ctx_obs_valid_batch = ctx_obs_valid_batch.bool().to(self._device)

                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)


                # Make context and target
                target_X = (tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                context_X = torch.cat([ctx_obs_batch, ctx_act_batch], dim=-1)
                context_Y = ctx_target_batch



                # Forward Pass
                out_mean, out_var, mu_z, cov_z = self._model(context_X, context_Y, target_X)

                self._te_sample_gt = tar_tar_batch.detach().cpu().numpy()
                self._te_sample_valid = tar_obs_valid_batch.detach().cpu().numpy()
                self._te_sample_pred_mu = out_mean.detach().cpu().numpy()
                self._te_sample_pred_var = out_var.detach().cpu().numpy()

                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(tar_tar_batch, out_mean, out_var)
                else:
                    loss = mse(tar_tar_batch, out_mean)

                metric_nll = gaussian_nll(tar_tar_batch, out_mean, out_var)
                metric_mse = mse(tar_tar_batch, out_mean)
#
                z_vis_list.append(mu_z.detach().cpu().numpy())
                task_id_list.append(task_id.detach().cpu().numpy())

                avg_loss += loss.detach().cpu().numpy()
                avg_metric_nll += metric_nll.detach().cpu().numpy()
                avg_metric_mse += metric_mse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        avg_metric_nll = avg_metric_nll / len(list(loader))
        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))
        z_vis = np.concatenate(z_vis_list, axis=0)
        #z_vis = 0
        task_labels = np.concatenate(task_id_list, axis=0)
        return avg_loss, avg_metric_nll, avg_metric_rmse, z_vis, task_labels

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor,
              train_targets: torch.Tensor, train_task_idx: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_task_idx: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
        '''
        :param train_obs: training observations for the model (includes context and targets)
        :param train_act: training actions for the model (includes context and targets)
        :param train_targets: training targets for the model (includes context and targets)
        :param train_task_idx: task_index for different training sequence
        :param epochs: number of epochs to train on
        :param batch_size: batch_size for gradient descent
        :param val_obs: validation observations for the model (includes context and targets)
        :param val_act: validation actions for the model (includes context and targets)
        :param val_targets: validation targets for the model (includes context and targets)
        :param val_task_idx: task_index for different testing sequence
        :param val_interval: how often to perform validation
        :param val_batch_size: batch_size while performing inference
        :return:
        '''

        """ Train Loop"""
        torch.cuda.empty_cache() #### Empty Cache
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf
        best_nll = np.inf
        best_rmse = np.inf

        if self._log:
            wandb.watch(self._model, log='all')
            artifact = wandb.Artifact('saved_model', type='model')

        for i in range(epochs):
            if i == 0:
                print('<<<<<<<<<<<<Set Encoder:>>>>>>>>>>>>', self.c.set_encoder)
                print('<<<<<<<<<<<<Neural Process:>>>>>>>>>>>>', self.c.hiprssm)
                print('<<<<<<<<<<<<SSM:>>>>>>>>>>>>', self.c.ssm_decoder)
            train_loss, train_metric_nll, train_metric_rmse, z_vis, z_labels, time = self.train_step(train_obs,
                                                                                                     train_act,
                                                                                                     train_targets,
                                                                                                     train_task_idx,
                                                                                                     batch_size)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(
                i + 1, self._loss, train_loss, 'target_nll:', train_metric_nll, 'target_rmse:', train_metric_rmse,
                time))
            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll,
                           "rmse/train_metric": train_metric_rmse, "epochs": i})

            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric_nll, val_metric_rmse, z_vis_val, z_labels_val = self.eval(val_obs, val_act,
                                                                                               val_targets,
                                                                                               val_task_idx,
                                                                                               batch_size=val_batch_size)
                if val_loss < best_loss:
                    if self.save_model:
                        print('>>>>>>>Saving Best Model<<<<<<<<<<')
                        torch.save(self._model.state_dict(), self._save_path)
                    if self._log:
                        wandb.run.summary['best_loss'] = val_loss
                    best_loss = val_loss
                if val_metric_nll < best_nll:
                    if self._log:
                        wandb.run.summary['best_nll'] = val_metric_nll
                    best_nll = val_metric_nll
                if val_metric_rmse < best_rmse:
                    if self._log:
                        wandb.run.summary['best_rmse'] = val_metric_rmse
                    best_rmse = val_metric_rmse
                print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll',
                                                                              val_metric_nll, 'target_rmse',
                                                                              val_metric_rmse))

                if self._log:
                    wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll,
                               "rmse/test_metric": val_metric_rmse, "epochs": i})

                if self._cluster_vis:
                    z_vis_concat = np.concatenate((z_vis, z_vis_val), axis=0)
                    z_labels_concat = np.concatenate((z_labels, z_labels_val), axis=0)
                    ####### Visualize the tsne embedding of the latent space in tensorboard
                    if self._log and i == (epochs - 1):
                        print('>>>>>>>>>>>>>Visualizing Latent Space<<<<<<<<<<<<<<', '---Epoch----', i)
                        ind = np.random.permutation(z_vis.shape[0])
                        z_vis = z_vis[ind, :]
                        z_vis = z_vis[:2000, :]
                        z_labels = z_labels[ind]
                        z_labels = z_labels[:2000]
                        ####### Visualize the tsne/pca in matplotlib / pyplot

                        plot_clustering(z_vis_concat, z_labels_concat, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis, z_labels, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)
                        plot_clustering(z_vis_val, z_labels_val, engine='matplotlib',
                                        exp_name=self._exp_name + '_' + str(i), wandb_run=self._run)

        if self.c.learn.save_model:
            artifact.add_file(self._save_path)
            wandb.log_artifact(artifact)
        if self.c.learn.plot_traj:
            plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,
                           self._run, log_name='train', exp_name=self._exp_name)
            plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,
                           self._run, log_name='test', exp_name=self._exp_name)

