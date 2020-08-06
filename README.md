
# Efficient-RL project
This is the code for a project lead for an internship with the [Br.A.In (Brain Inspired Artificial Intelligence)](http://recherche.telecom-bretagne.eu/brain/) research team in IMT Atlantique.   
We aim to use recent advances in unsupervised and self-supervised  to make current Reinforcement Learning algorithms more efficient.  

The code was forked from RL Baselines3 Zoo, for which you can find the documentation for on the [original repo](https://github.com/DLR-RM/rl-baselines3-zoo).

Contributions : 

|  RL Algo |  Implementation         | Tuning with Optuna         | Training with Zoo        |  Enjoy with Zoo |
|----------|--------------------|--------------------|--------------------|-------|
| Custom PPO      |:heavy_check_mark:| :heavy_check_mark:|:heavy_check_mark:| :heavy_check_mark: |

Custom algorithm code is contained in /ppo_representation

## Install :

`pip install -r requirements.txt`

## Sample usage guide 
The algorithm's name is ppo_representation, you can use features of baseline Zoo with that name.  
You can find some bash scripts in the scripts folder
### Tuning our algorithm
`python3 train.py --algo ppo_representation --env CartPole-v1  -optimize --n-trials 100 --n-jobs 12 
`

### Train our algorithm
`python3 train.py --algo ppo_representation --env MountainCarContinuous-v0 -tb ./runs/ppo_representation/tuned
`
