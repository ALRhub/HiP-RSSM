# Hidden Parameter Recurrent State Space Models (HiP-RSSM)
Pytorch code for ICLR 2022 paper [Hidden Parameter Recurrent State Space Models For Changing Dynamics Scenarios](https://openreview.net/forum?id=ds8yZOUsea). The algorithm learns
deep multi task Kalman Filters that can be used in non-stationary environments with changing dynamics.

<img src="/pics/img.png" alt="drawing" style="width:300px;"/>

Dependencies
--------------

* torch==1.3.1
* python 3.7
* omegaconf==2.1.1
* hydra-core==1.1.1
* PyYAML==5.3
* wandb==0.10.25
* umap-learn

How to Train
-------------

With ```HiP-RSSM``` as the working directory execute the python script
```python experiments/mobileRobot/mobile_robot_hiprssm.py model=default```


Datasets
------------
The dataset used here is that of a mobile robot traversing terrain of different slopes as reported in the paper. 

For Experimenting With New Datasets
-------------
For any dataset with a long timeseries, split them to reasonable local trajectories of length L=2*K, which is 
fed into the hiprssm model.
The first K would used by context encoder to infer latent context and the last K would be used as target set.
The concept is very similar to context sets and target sets in [Neural Processes](https://arxiv.org/abs/1807.01622) or the meta testing procedure used
in this [reference](https://openreview.net/pdf?id=HyztsoC5Y7).

A detailed description for creating training datasets is given in Appendix E.
A detailed description for testtime inference procedure is given in Algorithm 1 in the appendix.
![Alt text](/pics/img_1.png?raw=true "Test time inference")

How To Run Baselines
-------------
With ```HiP-RSSM``` as the working directory execute the python script
* LSTM Baseline:
```python experiments/mobileRobot/mobile_robot_rnn.py model=default_lstm```
* GRU Baseline:
```python experiments/mobileRobot/mobile_robot_rnn.py model=default_gru```
* RKN Baseline:
```python experiments/mobileRobot/mobile_robot_hiprssm.py model=default_rkn```