from omegaconf import DictConfig, OmegaConf
import hydra
import sys
sys.path.append('.')

import torch

from data.mobileDataSeq import metaMobileData
from metaWorldModels.ssm.LSTMBaseline import LSTMBaseline
from learning.lstm_trainer import Learn
from inference.lstm_inference import Infer
from utils.metrics import root_mean_squared
import wandb
nn = torch.nn


class mobileLSTM(LSTMBaseline):

    def __init__(self, obs_dim: int, act_dim: int, target_dim: int, lod: int, cell_config: OmegaConf,
                 layer_norm: bool,
                 use_cuda_if_available: bool = True):
        self._layer_norm = layer_norm
        super(mobileLSTM, self).__init__(obs_dim, act_dim, target_dim, lod, cell_config, use_cuda_if_available)

    def _build_obs_hidden_layers(self):
        layers = []
        last_hidden = self._obs_dim
        # hidden layers
        for hidden_dim in self.c.enc_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_act_hidden_layers(self):
        layers = []
        last_hidden = self._act_dim
        # hidden layers
        for hidden_dim in self.c.enc_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_enc_hidden_layers(self):
        layers = []
        last_hidden = 2 * self._lod
        # hidden layers
        for hidden_dim in self.c.enc_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

    def _build_dec_hidden_layers(self):
        layers = []
        last_hidden = 5 * self._lod
        # hidden layers
        for hidden_dim in self.c.dec_hidden_units:
            layers.append(nn.Linear(in_features=last_hidden, out_features=hidden_dim))
            layers.append(nn.ReLU())
            last_hidden = hidden_dim
        return nn.ModuleList(layers), last_hidden

def generate_mobile_robot_data_set(data):
    train_windows, test_windows = data.train_windows, data.test_windows

    train_targets = train_windows['target']
    test_targets = test_windows['target']

    train_obs = train_windows['obs']
    test_obs = test_windows['obs']

    train_task_idx = train_windows['task_index']
    test_task_idx = test_windows['task_index']

    train_act = train_windows['act']
    test_act = test_windows['act']

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()



@hydra.main(config_path='conf',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    config = cfg
    exp = Experiment(config)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg.model
        self._experiment()

    def _experiment(self):
        cfg = self.global_cfg
        torch.cuda.empty_cache()

        data = metaMobileData(cfg.data_reader)
        train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(
            data)  # If your dataset do not have actions you can set them as zero for now, I will update the code for datasets with unactuated dynamcis

        ####
        impu = cfg.data_reader.imp

        ##### Define WandB Stuffs
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(project=cfg.wandb.project_name, name=expName,
                               mode=mode)  # wandb object has a set of configs associated with it as well


        ##### Define Model, Train and Inference Modules

        lstm_model = mobileLSTM(obs_dim=train_obs.shape[-1], act_dim= train_act.shape[-1], target_dim= train_targets.shape[-1], lod=cfg.lstm.latent_obs_dim, cell_config=cfg.lstm, layer_norm=True)
        lstm_learn = Learn(lstm_model, loss='mse', half_sequence=True, imp=impu, config=cfg, run=wandb_run, log=cfg.wandb['log'])


        if cfg.learn.load == False:
            #### Train the Model
            lstm_learn.train(train_obs, train_act, train_targets, cfg.learn.epochs, cfg.learn.batch_size, test_obs, test_act,
                              test_targets)




        ########################################## Inference And Testing Multi Step Ahead Predictions#################################################
        ##### Load best model
        lstm_model.load_state_dict(torch.load(lstm_learn._save_path))

        ###### Inference

        tar_type = cfg.data_reader.tar_type

        ##########  Initialize inference class
        lstm_infer = Infer(lstm_model,  normalizer=data.normalizer, config=cfg, run=wandb_run)
        batch_size = 10
        k = int(train_obs.shape[1] / 2)
        pred_mean, pred_var, gt, _, _ = lstm_infer.predict(test_obs, test_act, test_targets,
                                                                    imp=impu, k=k,
                                                                    test_gt_known=True, batch_size=batch_size, tar=tar_type)
        print(pred_mean.shape, pred_var.shape, gt.shape)
        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data.normalizer, tar="observations", denorma=True)
        print(rmse_next_state)
        wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state

        multiSteps = [1,3,5,10,20,30,40,50]
        for step in multiSteps:
            pred_mean, pred_var, gt = lstm_infer.predict_mbrl(test_obs, test_act, test_targets, k=k,
                                                            batch_size=batch_size,
                                                            multiStep=step, tar=tar_type)
            rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data.normalizer, tar="observations", denorma=True)
            wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state
            print(rmse_next_state)



def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()