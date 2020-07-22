import gym

from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage

from ppo_representation import PPORepresentation

from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import difflib
import os
import importlib
import time
import uuid
import warnings
from collections import OrderedDict
from pprint import pprint

import yaml
import gym
import seaborn
import numpy as np
import torch as th
# For custom activation fn
import torch.nn as nn  # noqa: F401 pytype: disable=unused-import

from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class
from utils.hyperparams_opt import hyperparam_optimization
from utils.callbacks import SaveVecNormalizeCallback
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict, get_callback_class

seaborn.set()

def create_env(n_envs, eval_env=False, no_log=False):
    """
    Create the environment and wrap it if necessary
    :param n_envs: (int)
    :param eval_env: (bool) Whether is it an environment used for evaluation or not
    :param no_log: (bool) Do not log training when doing hyperparameter optim
        (issue with writing the same file)
    :return: (Union[gym.Env, VecEnv])
    """
    global hyperparams
    global env_kwargs

    # Do not log eval env (issue with writing the same file)
    log_dir = None if eval_env or no_log else save_path

    if n_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, args.seed,
                                    wrapper_class=env_wrapper, log_dir=log_dir,
                                    env_kwargs=env_kwargs)])
    else:
        # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = DummyVecEnv([make_env(env_id, i, args.seed, log_dir=log_dir, env_kwargs=env_kwargs,
                                    wrapper_class=env_wrapper) for i in range(n_envs)])
    if normalize:
        # Copy to avoid changing default values by reference
        local_normalize_kwargs = normalize_kwargs.copy()
        # Do not normalize reward for env used for evaluation
        if eval_env:
            if len(local_normalize_kwargs) > 0:
                local_normalize_kwargs['norm_reward'] = False
            else:
                local_normalize_kwargs = {'norm_reward': False}

        if args.verbose > 0:
            if len(local_normalize_kwargs) > 0:
                print(f"Normalization activated: {local_normalize_kwargs}")
            else:
                print("Normalizing input and reward")
        env = VecNormalize(env, **local_normalize_kwargs)

    # Optional Frame-stacking
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print(f"Stacking {n_stack} frames")

    if is_image_space(env.observation_space):
        if args.verbose > 0:
            print("Wrapping into a VecTransposeImage")
        env = VecTransposeImage(env)
    return env

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func
# def train(policy, env, learning_rate, n_steps, batch_size,
#           nepochs,gamma, gae_lambda, clip_range, ent_coef,
#           max_grad_norm,use_sde, sde_sample_freq, target_kl):
#     """Train a policy on the given env """
#
#
#
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    env = DummyVecEnv()

    model = PPORepresentation('MlpPolicy', env, verbose = 1,tensorboard_log= "runs/Representation/CartPole-v1",n_steps= 32,
                              batch_size= 256, gae_lambda= 0.8,gamma= 0.98, n_epochs= 20,ent_coef= 0,
                              learning_rate= linear_schedule(0.001),clip_range= linear_schedule(0.2), )
    model.learn(total_timesteps= 1000000)



# CartPole-v1:
#   n_envs: 8
#   n_timesteps: !!float 1e5
#   policy: 'MlpPolicy'
#   n_steps: 32
#   batch_size: 256
#   gae_lambda: 0.8
#   gamma: 0.98
#   n_epochs: 20
#   ent_coef: 0.0
#   learning_rate: lin_0.001
#   clip_range: lin_0.2