import sys
sys.path.append('.')
from omegaconf import DictConfig, OmegaConf
import hydra
import os

import torch
import wandb

from data.mobileDataSeq import metaMobileData
from metaWorldModels.hiprssm import HipRSSM
from learning import hiprssm_dyn_trainer
from inference import hiprssm_dyn_inference
from utils.metrics import root_mean_squared

nn = torch.nn


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
    print(test_act.shape, train_act.shape)

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(train_task_idx).float(),\
           torch.from_numpy(test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_task_idx).float()

@hydra.main(config_path='conf',config_name='config')
def my_app(cfg)->OmegaConf:
    global config
    model_cfg = cfg.model
    exp = Experiment(model_cfg)


class Experiment():
    def __init__(self, cfg):
        self.global_cfg = cfg
        self._experiment()


    def _experiment(self):
        """Data"""
        cfg = self.global_cfg
        torch.cuda.empty_cache()

        tar_type = cfg.data_reader.tar_type  # 'delta' - if to train on differences to current states
        # 'next_state' - if to trian directly on the  next states

        ### Load Data Here
        data = metaMobileData(cfg.data_reader)
        train_obs, train_act, train_targets, train_task_idx, test_obs, test_act, test_targets, test_task_idx = generate_mobile_robot_data_set(
            data) # If your dataset do not have actions you can set them as zero for now, I will update the code for datasets with unactuated dynamcis

        ####
        impu = cfg.data_reader.imp
        save_path = os.getcwd() + '/experiments/saved_models/' + cfg.wandb.exp_name + '.ckpt'

        ##### Define WandB Stuffs
        expName = cfg.wandb.exp_name
        if cfg.wandb.log:
            mode = "online"
        else:
            mode = "disabled"

        ## Initializing wandb object and sweep object
        wandb_run = wandb.init(project=cfg.wandb.project_name, name=expName,
                               mode=mode)  # wandb object has a set of configs associated with it as well

        ### Initialize Model Classes, Train and Inference Modules
        hiprssm_model = HipRSSM(obs_dim=train_obs.shape[-1], action_dim=train_act.shape[-1],
                          target_dim=train_targets.shape[-1],
                          config=cfg)


        hiprssm_learn = hiprssm_dyn_trainer.Learn(hiprssm_model, loss=cfg.learn.loss, imp=impu, config=cfg, run=wandb_run,
                                           log=cfg.wandb['log'])

        if cfg.learn.load == False:
            #### Train the Model
            hiprssm_learn.train(train_obs, train_act, train_targets, train_task_idx, cfg.learn.epochs, cfg.learn.batch_size,
                           test_obs, test_act,
                           test_targets, test_task_idx)




        ########################################## Inference And Testing Multi Step Ahead Predictions#################################################
        ##### Load best model
        model_at = wandb_run.use_artifact('saved_model' + ':latest')
        model_path = model_at.download()  ###return the save durectory path in wandb local
        hiprssm_model.load_state_dict(torch.load(save_path))
        print('>>>>>>>>>>Loaded The Model From Local Folder<<<<<<<<<<<<<<<<<<<')

        ###### Inference

        ##########  Initialize inference class
        hiprssm_infer = hiprssm_dyn_inference.Infer(hiprssm_model, normalizer=data.normalizer, config=cfg, run=wandb_run)
        batch_size = 2
        k = int(train_obs.shape[1] / 2)
        pred_mean, pred_var, gt, obs_valid, _, _, cur_obs = hiprssm_infer.predict(test_obs, test_act, test_targets, test_task_idx,
                                                                    imp=impu, k=k,
                                                                    test_gt_known=True, batch_size=batch_size, tar=tar_type)
        print(pred_mean.shape, pred_var.shape, gt.shape, obs_valid.shape)



        rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt, data.normalizer,
                                                                  tar="observations", denorma=True)
        wandb_run.summary['rmse_denorma_next_state'] = rmse_next_state

        print("Root mean square Error is:", rmse_next_state)


        multiSteps = [1,50, 100, 120]
        for step in multiSteps:
             pred_mean, pred_var, gt_multi = hiprssm_infer.predict_mbrl(test_obs, test_act, test_targets, k=k,
                                                             batch_size=batch_size,
                                                             multiStep=step, tar=tar_type)

             rmse_next_state, pred_obs, gt_obs = root_mean_squared(pred_mean, gt_multi, data.normalizer, tar="observations", denorma=True)
             print(step,rmse_next_state)
             wandb_run.summary['rmse_multi_step_' + str(step)] = rmse_next_state


def main():
    my_app()



## https://stackoverflow.com/questions/32761999/how-to-pass-an-entire-list-as-command-line-argument-in-python/32763023
if __name__ == '__main__':
    main()