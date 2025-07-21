# Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation

This repo is the official project repository of [\[**LE-Nav**\]](https://arxiv.org/pdf/2507.11001) ([\[DEMO\]](https://drive.google.com/file/d/1_XVsA-nbONcEre_OyEVM9BInMulYK7_r/view?usp=sharing)).

<p align="center">
  <img src="/fig/scene1.gif" width="200"/>
  <img src="/fig/scene2.gif" width="200"/>
  <img src="/fig/scene3.gif" width="200"/>
  <img src="/fig/scene4.gif" width="200"/>
</p>

<div align="left">

## 1. Overview
![image](fig/comparison.jpg)

LE-Nav is an interpretable and adaptive navigation framework designed for service robots operating in dynamic, human-centric environments. Traditional navigation systems often struggle in such unstructured settings due to fixed parameters and poor generalization. LE-Nav addresses this by combining multi-modal large language models (MLLMs) with conditional variational autoencoders (CVAEs) for zero-shot scene understanding and expert-level parameter tuning.

![image](fig/overview.jpg)

## 2. Environment
Download the code and create environment.
```
conda env create -f environment.yml
```
You can also try:
```
conda create --name readscene python=3.9
conda activate readscene
```
Install dependencies.
```
pip install openai 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy==1.22.4
conda install tensorboard
pip install ultralytics
```
or
```
pip install -r requirements.txt
```
## 3. Training
Collect the data for your planner. Customize your config.yaml.
```
python train_cvae.py
```
In our case, we select the following eight key hyperparameters for TEB: max_vel_x, max_vel_theta, acc_lim_x, acc_lim_theta, weight_max_vel_x, weight_acc_lim_x, weight_acc_lim_theta, weight_optimaltime
and eight key hyperparameters for DWA: max_vel_x, max_vel_theta, acc_lim_x, acc_lim_theta, path_distance_bias, goal_distance_bias, occdist_scale, forward_point_distance. In the case of DWA, when updating hyperparameter max_vel_x,
we additionally synchronize the value of max_vel_trans to be consistent with max_vel_x. During the deployment, the inflation_radius of global costmap is also recorded and learned as it is closely related to the local planner.

## 4. Deployment
Fill in the path, api key in the ROS file.
```
source ~/your_ws/devel/setup.bash
rosrun your_package path/to/image_infer_node.py
```
If you use the same setup, you can try the [\[model parameters\]](https://github.com/Cavendish518/LE-Nav/tree/main/weight) we provide.
## 5. Citation
If your like our projects, please cite us and give this repo a star.
```
@article{wang2025learning,
  title={Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation},
  author={Wang, Yanbo and Fang, Zipeng and Zhao, Lei and Chen, Weidong},
  journal={arXiv preprint arXiv:2507.11001},
  year={2025}
}
```
