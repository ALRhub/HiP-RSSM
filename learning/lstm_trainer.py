import os
import time as t
from typing import Tuple
import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

from metaWorldModels.ssm.LSTMBaseline import LSTMBaseline
from utils.Losses import mse, gaussian_nll
from utils.plotTrajectory import plotImputation
from utils.dataProcess import split_k_m, get_ctx_target_impute


optim = torch.optim
nn = torch.nn


class Learn:

    def __init__(self, model: LSTMBaseline, loss: str, imp: float = 0.0, half_sequence: bool=True , config=None, run=None, log: bool=True, use_cuda_if_available: bool = True):
        """
        :param model: nn module for rkn
        :param loss: type of loss to train on 'nll' or 'mse'
        :param metric: type of metric to print during training 'nll' or 'mse'
        :param use_cuda_if_available: if gpu training set to True
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._loss = loss
        self._model = model
        self._imp = imp
        if config is None:
            raise TypeError('Pass a Config Dict')
        else:
            self.c = config
        self._learning_rate = self.c.learn.lr
        self._save_path = os.getcwd() + '/experiments/saved_models/' + run.name + '.ckpt'
        self._exp_name = run.name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches
        self._half_sequence = half_sequence
        self._log = bool(log)
        if self._log:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', type(self._log), self._log)
            self._run = run

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray,
                   train_targets: np.ndarray, batch_size: int) \
            -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_targets: training targets
        :param batch_size:
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        avg_loss = avg_metric_nll = avg_metric_mse = 0
        t0 = t.time()
        b = list(loader)[0]

        for batch_idx, (obs, act, targets) in enumerate(loader):
            # Assign tensors to devices
            obs = (obs).to(self._device)
            act = act.to(self._device)
            targets = (targets).to(self._device)

            if self._half_sequence:
                # Split to context and targets
                k = int(obs.shape[1] / 2)
                m = obs.shape[1] - k
            else:
                k = 0
                m = ctx_obs_batch.shape[1]

            ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                get_ctx_target_impute(obs, act, targets, k, num_context=None, tar_imp=self._imp,
                                      random_seed=True)

            tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)


            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            out_mean, out_var = self._model(tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

            ## Calculate Loss
            if self._loss == 'nll':
                loss = gaussian_nll(tar_tar_batch, out_mean, out_var)
            else:
                loss = mse(tar_tar_batch, out_mean)

            # Backward Pass
            loss.backward()

            # Clip Gradients
            if self.c.lstm.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            with torch.no_grad():
                # Calculate metrics only on targets for fair comparison
                metric_nll = gaussian_nll(tar_tar_batch, out_mean, out_var)
                metric_mse = mse(tar_tar_batch, out_mean)



            avg_loss += loss.detach().cpu().numpy()
            avg_metric_nll += metric_nll.detach().cpu().numpy()
            avg_metric_mse += metric_mse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        with torch.no_grad():
            self._tr_sample_gt = tar_tar_batch.cpu().numpy()
            self._tr_sample_valid = tar_obs_valid_batch.cpu().numpy()
            self._tr_sample_pred_mu = out_mean.cpu().numpy()
            self._tr_sample_pred_var = out_var.cpu().numpy()

        avg_metric_nll = avg_metric_nll / len(list(loader))

        avg_metric_rmse = np.sqrt(avg_metric_mse / len(list(loader)))

        return avg_loss, avg_metric_nll, avg_metric_rmse, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, targets: np.ndarray,
             batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evalauate on
        :param obs_valid: observation valid flag
        :param targets: targets to evaluate on
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        dataset = TensorDataset(obs, act, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        avg_loss = 0.0
        avg_metric_nll = avg_metric_rmse = 0


        for batch_idx, (obs, act, targets) in enumerate(loader):
            # Assign tensors to devices
            obs = (obs).to(self._device)
            act = act.to(self._device)
            targets = (targets).to(self._device)

            with torch.no_grad():
                # Assign tensors to devices
                if self._half_sequence:
                    # Split to context and targets
                    k = int(obs.shape[1] / 2)
                    m = obs.shape[1] - k
                else:
                    k = 0
                    m = ctx_obs_batch.shape[1]

                ctx_obs_batch, ctx_act_batch, ctx_target_batch, tar_obs_batch, tar_act_batch, tar_tar_batch, tar_obs_valid_batch = \
                    get_ctx_target_impute(obs, act, targets, k, num_context=None, tar_imp=self._imp,
                                          random_seed=True)

                tar_obs_valid_batch = torch.from_numpy(tar_obs_valid_batch).bool().to(self._device)
                # Forward Pass
                out_mean, out_var = self._model(tar_obs_batch, tar_act_batch, tar_obs_valid_batch)

                self._te_sample_gt = tar_tar_batch.cpu().numpy()
                self._te_sample_valid = tar_obs_valid_batch.cpu().numpy()
                self._te_sample_pred_mu = out_mean.cpu().numpy()
                self._te_sample_pred_var = out_var.cpu().numpy()

                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(tar_tar_batch, out_mean, out_var)
                else:
                    loss = mse(tar_tar_batch, out_mean)




                # Calculate metrics only on targets for fair comparison

                metric_nll = gaussian_nll(tar_tar_batch, out_mean, out_var)
                metric_rmse = mse(tar_tar_batch, out_mean)

                avg_loss += loss.detach().cpu().numpy()
                avg_metric_nll += metric_nll.detach().cpu().numpy()
                avg_metric_rmse += metric_rmse.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        avg_metric_nll = avg_metric_nll / len(list(loader))

        avg_metric_rmse = np.sqrt(avg_metric_rmse / len(list(loader)))

        return avg_loss, avg_metric_nll, avg_metric_rmse

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor,
              train_targets: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
        """
        Train function
        :param train_obs: observations for training
        :param train_targets: targets for training
        :param epochs: number of epochs to train for
        :param batch_size: batch size for training
        :param val_obs: observations for validation
        :param val_targets: targets for validation
        :param val_interval: validate every <this> iterations
        :param val_batch_size: batch size for validation, to save memory
        """

        """ Train Loop"""
        print('<<<<<<<<<<<<AcRKNr:>>>>>>>>>>>>', self.c.items())
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf
        best_nll = np.inf
        best_rmse = np.inf
        if self._log:
            wandb.watch(self._model,log=all)
            artifact = wandb.Artifact('model',type='model')

        for i in range(epochs):
            train_loss, train_metric_nll, train_metric_rmse, time = self.train_step(train_obs, train_act, train_targets,
                                                             batch_size)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(
                i + 1, self._loss, train_loss, 'target_nll:', train_metric_nll, 'target_rmse:', train_metric_rmse, time))
            # self._writer.add_scalar(self._loss + "/train_loss", train_loss, i)
            # self._writer.add_scalar("nll/train_metric", train_metric_nll, i)
            # self._writer.add_scalar("rmse/train_metric", train_metric_rmse, i)
            if self._log:
                wandb.log({self._loss + "/train_loss": train_loss, "nll/train_metric": train_metric_nll,
                           "rmse/train_metric": train_metric_rmse, "epochs": i})
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric_nll, val_metric_rmse = self.eval(val_obs, val_act, val_targets,
                                                 batch_size=val_batch_size)
                if val_loss<best_loss:
                    print('>>>>>>>Saving Best Model<<<<<<<<<<')
                    torch.save(self._model.state_dict(), self._save_path)
                    if self._log:
                        wandb.run.summary['best_loss']=val_loss
                    best_loss = val_loss
                if val_metric_nll<best_nll:
                    if self._log:
                        wandb.run.summary['best_nll']=val_metric_nll
                    best_nll = val_metric_nll
                if val_metric_rmse<best_rmse:
                    if self._log:
                        wandb.run.summary['best_rmse']=val_metric_rmse
                    best_rmse = val_metric_rmse
                print("Validation: {}: {:.5f}, {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, 'target_nll', val_metric_nll, 'target_rmse', val_metric_rmse))
                # self._writer.add_scalar(self._loss+"/test_loss", val_loss, i)
                # self._writer.add_scalar("nll/test_metric", val_metric_nll, i)
                # self._writer.add_scalar("rmse/test_metric", val_metric_rmse, i)
                if self._log:
                    wandb.log({self._loss + "/val_loss": val_loss, "nll/test_metric": val_metric_nll,
                           "rmse/test_metric": val_metric_rmse, "epochs": i})
        if self._log:
            plotImputation(self._tr_sample_gt, self._tr_sample_valid, self._tr_sample_pred_mu, self._tr_sample_pred_var,
                       self._run, log_name='train', exp_name=self._exp_name)
            plotImputation(self._te_sample_gt, self._te_sample_valid, self._te_sample_pred_mu, self._te_sample_pred_var,
                       self._run, log_name='test', exp_name=self._exp_name)
            artifact.add_file(self._save_path)
            wandb.log_artifact(artifact)


