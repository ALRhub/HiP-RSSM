from matplotlib import pyplot as plt
import numpy as np

from utils.dataProcess import norm, denorm

def root_mean_squared(pred, target, normalizer=None, tar='observations', fromStep=0, denorma=False, plot=None):
    """
    root mean squared error
    :param target: ground truth positions
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: root mean squared error between targets and predicted mean, predicted variance is ignored
    """
    if type(pred) is not np.ndarray:
        pred = pred.cpu().detach().numpy()
    if type(target) is not np.ndarray:
        target = target.cpu().detach().numpy()
    pred = pred[..., :target.shape[-1]]

    sumSquare = 0
    count = 0
    if plot != None:
        for idx in range(target.shape[2]):
            plt.plot(target[3,:,idx],label='target')
            plt.plot(pred[3,:,idx],label='prediction')
            plt.legend()
            plt.show()

    if denorma==True:
        pred = denorm(pred, normalizer, tar)
        target = denorm(target, normalizer, tar)



    #target = target[:, fromStep:, :]
   # pred = pred[:, fromStep:, :]
    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    #print('RMSE Samplesss......................................',numSamples)
    sumSquare = np.sum(np.sum(np.sum((target - pred) ** 2)))
    return np.sqrt(sumSquare / numSamples), pred, target

def joint_mse(pred, target, data=[], tar='observations', fromStep=0, denorma=False, plot=None):
    """
    :return: mse
    """
    if denorma==True:
        pred = denorm(pred, data, tar)
        target = denorm(target, data, tar)

    numSamples = 1
    for dim in target.shape:
        numSamples = numSamples * dim
    # print('RMSE Samplesss......................................',numSamples)
    #sumSquare = np.sum(np.sum(((target - pred)/target) ** 2,0),0)
    sumSquare = np.sum(np.sum(((target - pred)) ** 2, 0), 0)
    return sumSquare / numSamples

# loss functions
def gaussian_nll(pred,target):
    """
    gaussian nll
    :param target: ground truth positions
    :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
    :return: gaussian negative log-likelihood
    """
    pred_mean, pred_var = pred[..., :target.shape[-1]], pred[..., target.shape[-1]:]

    pred_var += 1e-8
    element_wise_nll = 0.5 * (np.log(2 * np.pi) + np.log(pred_var) + ((target - pred_mean)**2) / pred_var)
    sample_wise_error = np.sum(element_wise_nll, axis=-1)
    return np.mean(sample_wise_error)


def comparison_plot(target,pred_list=[],name_list=[],data=[], tar='observations', denorma=False):
    '''
    :param target: ground truth
    :param pred_list: list of predictions to compare
    :param name_list: names to each of predictions given as a list
    :return:
    '''
    sample = np.random.randint(target.shape[0])
    sample = 0
    print('sample number',sample)
    #fig, axes = plt.subplots(5, sharex=True, sharey=True)
    if denorma==True:
        target = denorm(target, data, tar)
        for idx,pred in enumerate(pred_list):
            pred_list[idx] = denorm(pred, data, tar)
        #plt.ylim((-1, 1))
        # plt.legend()
        # plt.show()

    fig, axs = plt.subplots(3)
    for k,idx in enumerate([0,1,4]):
        axs[k].plot(target[sample, :, idx], label='GT')
        for pred,name in zip(pred_list,name_list):
            axs[k].plot(pred[sample, :, idx], label=name)
            axs[0].title.set_text('Torque Preditctions For Joint 1, 4 and 5')
            axs[k].legend()
            axs[k].set(ylabel="Torque(Nm)")
        #plt.ylim((-1
        # , 1))
        #plt.legend()
    plt.show()

def naive_baseline(current_obs,targets,data=[],tar_type='observations',steps=[1,3,5,10,20],denorma=False):
    '''
    :param current_obs: current available observations
    :param targets: actual targets
    :param steps: list of steps for calculating n step ahead prediction accuracy
    :return: Nothing
    '''
    if type(current_obs) is not np.ndarray:
        current_obs = current_obs.cpu().detach().numpy()
    if type(targets) is not np.ndarray:
        targets = targets.cpu().detach().numpy()

    for step in steps:
        if step==1:
            pred=current_obs
        else:
            pred = current_obs[:,:-(step-1),:]
        tar = targets[:,step-1:,:]
        print('root mean square error step',step,root_mean_squared(pred,tar,data,tar=tar_type,denorma=denorma)[0])

