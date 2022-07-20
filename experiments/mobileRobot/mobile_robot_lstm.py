from omegaconf import DictConfig, OmegaConf
import hydra
import sys
sys.path.append('.')
import os

import numpy as np
import torch


from data.mobileDataSeq import metaMobileData
from data.mobileDataSeq_Infer import metaMobileDataInfer
from dynamics_models.rkn.acrkn.LSTMBaseline import LSTMBaseline
from learning.lstm_trainer import Learn
from inference.lstm_inference import Infer
from utils.metrics import naive_baseline
from utils.metrics import root_mean_squared
from utils.multistepRecurrent import longHorizon_Seq
from utils.ConfigDict import ConfigDict
from utils.dataProcess import split_k_m
import wandb
nn = torch.nn


def generate_mobile_robot_data_set(data, dim):
    train_windows, test_windows = data.train_windows, data.test_windows

    train_targets = train_windows['target'][:,:,:dim]
    test_targets = test_windows['target'][:,:,:dim]

    train_obs = train_windows['obs'][:,:,:dim]
    test_obs = test_windows['obs'][:,:,:dim]

    train_task_idx = train_windows['task_index']
    test_task_idx = test_windows['task_index']

    train_act = train_windows['act'][:,:,:dim]
    test_act = test_windows['act'][:,:,:dim]

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()



@hydra.main(config_path='conf',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    config = cfg
    exp = Experiment(config, sweep=False)


class Experiment():
    def __init__(self, cfg, sweep=False):
        self.global_cfg = cfg
        self.sweep = sweep
        self._setsweep = self._set_sweep()


    def _set_sweep(self):
        if self.sweep:
            sweep_conf = OmegaConf.to_container(self.global_cfg.sweep)
            sweep_id = wandb.sweep(sweep_conf)
            wandb.agent(sweep_id, self._experiment, count=1)
        else:
            self._experiment(None)

    def _align_global_sweep_conf(self, cfg, sweep_cfg):
        # align sweep config and global config
        cfg.learn.lr = sweep_cfg['learn.lr']
        cfg.np.latent_obs_dim = sweep_cfg['np.latent_obs_dim']
        return cfg


    def _experiment(self, sweep_conf=None):
        '''
        joints : Give a list of joints (0-6) on which you want to train on eg: [1,4]
        lr: Learning Rate
        num_basis : Number of transition matrices to be learnt
        '''
        cfg = self.global_cfg.model
        print(sweep_conf)


        #data = mpgRobotSeq(targets=tar_type, standardize=True, load=bool(int(data_type_load_name[1])), file_name=data_type_load_name[2])
        data = metaMobileData(cfg.data_reader)
        train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(
            data, cfg.data_reader.dim)

        act_dim = train_act.shape[-1]
        obs_dim = train_obs.shape[-1]

        impu = cfg.data_reader.imp
        ##### Naive Baseline
        naive_baseline(test_obs[:, :-1, :], test_obs[:, 1:, :], data, denorma=True)
        print(test_obs.shape, train_targets.shape)

        save_path = os.getcwd() + '/experiments/saved_models/' + cfg.wandb.exp_name + '.ckpt'

        class mobileLSTM(LSTMBaseline):

            def __init__(self, target_dim: int, lod: int, lad: int, cell_config: ConfigDict,
                         layer_norm: bool,
                         use_cuda_if_available: bool = True):
                self._layer_norm = layer_norm
                super(mobileLSTM, self).__init__(target_dim, lod, lad, cell_config, use_cuda_if_available)

            def _build_obs_hidden_layers(self):
                layers = []
                last_hidden = obs_dim
                # hidden layers
                for hidden_dim in cfg.lstm.enc_hidden_units:
                    layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
                    layers.append(nn.ReLU())
                    last_hidden = hidden_dim
                return nn.ModuleList(layers), last_hidden

            def _build_act_hidden_layers(self):
                layers = []
                last_hidden = act_dim
                # hidden layers
                for hidden_dim in cfg.lstm.enc_hidden_units:
                    layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
                    layers.append(nn.ReLU())
                    last_hidden = hidden_dim
                return nn.ModuleList(layers), last_hidden

            def _build_enc_hidden_layers(self):
                layers = []
                last_hidden = 2 * self._lod
                # hidden layers
                for hidden_dim in cfg.lstm.enc_hidden_units:
                    layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
                    layers.append(nn.ReLU())
                    last_hidden = hidden_dim
                return nn.ModuleList(layers), last_hidden

            def _build_dec_hidden_layers(self):
                layers = []
                last_hidden = 5 * self._lod
                # hidden layers
                for hidden_dim in cfg.lstm.dec_hidden_units:
                    layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
                    layers.append(nn.ReLU())
                    last_hidden = hidden_dim
                return nn.ModuleList(layers), last_hidden


        ##### Naive Baseline
        naive_baseline(test_obs[:, :-1, :], test_obs[:, 1:, :], data, denorma=True)
        print(test_obs.shape, train_targets.shape)

        save_path = os.getcwd() + '/experiments/saved_models/' + cfg.wandb.exp_name + '.ckpt'
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.learn.gpu

        ##### Define WandB Stuffs
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(config=sweep_conf, project=cfg.wandb.project_name, name=expName,
                               mode=mode)  # wandb object has a set of configs associated with it as well
        #
        sweep_cfg = wandb_run.config  # make sure whats being logged and whats being passed are the same

        # align sweep config and global config
        if self.sweep:
            cfg = self._align_global_sweep_conf(cfg, sweep_cfg)

        ##### Define Model, Train and Inference Modules

        lstm_model = mobileLSTM(obs_dim, cfg.lstm.latent_obs_dim, act_dim, cell_config=cfg.lstm, layer_norm=True)
        lstm_learn = Learn(lstm_model, loss='mse', half_sequence=True, imp=impu, config=cfg, run=wandb_run, log=cfg.wandb['log'])


        if cfg.learn.load == False:
            #### Train the Model
            lstm_learn.train(train_obs, train_act, train_targets, cfg.learn.epochs, cfg.learn.batch_size, test_obs, test_act,
                              test_targets)

        ##### Load best model
        lstm_model.load_state_dict(torch.load(lstm_learn._save_path))

        ###### Inference

        tar_type = cfg.data_reader.tar_type

        ##########  Initialize inference class
        lstm_infer = Infer(lstm_model,  data=data, config=cfg, run=wandb_run)
        batch_size = 10
        k = int(train_obs.shape[1] / 2)
        test_Y_context, test_Y_target = split_k_m(test_targets, k, 0)
        pred_mean, pred_var, gt, _, _ = lstm_infer.predict(test_obs, test_act, test_targets,
                                                                    imp=impu, k=k,
                                                                    test_gt_known=True, batch_size=batch_size, tar=tar_type)
        print(pred_mean.shape, pred_var.shape, gt.shape)
        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data, tar="observations", denorma=True)
        print(rmse_next_state)
        wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state

        multiSteps = [1,3,5,10,20,30,40,50]
        for step in multiSteps:
            pred_mean, pred_var, gt = lstm_infer.predict_mbrl(test_obs, test_act, test_targets, k=k,
                                                            batch_size=batch_size,
                                                            multiStep=step, tar=tar_type)
            rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data, tar="observations", denorma=True)
            wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state
            print(rmse_next_state)

        # plotImputation(gt_obs, None, pred_obs, None,
        #                wandb_run, log_name='test', exp_name=expName)

        infer_changing = False
        if infer_changing:
            data = metaMobileDataInfer(cfg.data_reader)
            train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(
                data, cfg.data_reader.dim)
            print(test_act.shape, test_task_idx.shape)
            pred_mean, pred_var, gt, obs_valid, z_vis, labels = lstm_infer.predict(test_obs, test_act, test_targets,
                                                                                 test_task_idx,
                                                                                 imp=0.0, k=k,
                                                                                 test_gt_known=True,
                                                                                 batch_size=batch_size)
            print(z_vis.shape, labels.shape)
            norm_labels = (labels - labels.mean()) / (labels.max() - labels.min())
            print(norm_labels)
            labels = norm_labels


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()