import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import wandb


def plotImputation(gts, valid_flags, pred_mus, pred_stds, wandb_run, dims=None, num_traj: int =2, log_name='test', exp_name='trial', show=False):
    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    if dims is None:
        dims = np.arange(gts.shape[-1])
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            if valid_flags is not None:
                valid_flag = valid_flags[traj,:,0]
            pred_mu = pred_mus[traj,:,dim]
            if pred_stds is not None:
                pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            if valid_flags is not None:
                plt.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],gt[np.logical_not(valid_flag)],facecolor='red',s=14)
            plt.plot(pred_mu, color='black')
            # if pred_stds is not None:
            #     plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'Imp_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()

def plotImputationDiff(gts, valid_flags, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3,4,5], num_traj: int =2, log_name='test', exp_name='trial', show=False):
    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            if valid_flags is not None:
                valid_flag = valid_flags[traj,:,0]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            if valid_flags is not None:
                plt.scatter(torch.arange(len(valid_flag))[np.logical_not(valid_flag)],gt[np.logical_not(valid_flag)],facecolor='red',s=14)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'Imp_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()

def plotLongTerm(gts, pred_mus, pred_stds, wandb_run, dims=[0], num_traj=2, log_name='test', exp_name='trial', show=False):
    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'MultiStep_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()


def plotMbrl(gts, pred_mus, pred_stds, wandb_run, dims=[0,1,2,3], num_traj=2, log_name='test', exp_name='trial', show=False):
    folder_name = os.getcwd() + '/experiments/pam/runs/latent_plots'
    trjs = np.random.randint(gts.shape[0],size=num_traj)
    for traj in trjs:
        for dim in dims:
            gt = gts[traj,:,dim]
            pred_mu = pred_mus[traj,:,dim]
            pred_std = pred_stds[traj,:,dim]
            plt.Figure()
            plt.plot(gt)
            plt.plot(pred_mu, color='black')
            plt.fill_between(np.arange(len(gt)), pred_mu - pred_std, pred_mu + pred_std, alpha=0.2, color='grey')
            if show == True:
                plt.show()
                plt.close()
            else:
                plt.savefig(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                image = plt.imread(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                if wandb_run is not None:
                    key = 'MBRL_Trajectory_' + str(traj) + '_dim_' + str(dim) +'_' + log_name
                    wandb_run.log({key: wandb.Image(image)})
                    os.remove(folder_name + "/traj_" + str(traj) + '_dim_' + str(dim) + exp_name + ".png")
                    plt.close()



if __name__ == '__main__':
    global ax
    gt = np.random.rand(10,50,1)
    pred = np.random.rand(10,50,1)
    std = np.random.uniform(low=0.01, high=0.1, size=(10,50,1))
    rs = np.random.RandomState(seed=23541)
    obs_valid = rs.rand(gt.shape[0], gt.shape[1], 1) < 1 - 0.5
    pred = np.random.rand(10, 50, 1)
    plotSimple(gt[1,:,0],obs_valid[1,:,0],pred[1,:,0],pred_std=std[1,:,0])
    plotMbrl(gt[1,:,0],pred[1,:,0],pred_std=std[1,:,0])