import pandas as pd
import gym, time
import numpy as np
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
import pylab as pl
import csv

parentDir   = os.path.dirname(os.getpwd())
sys.path.insert(0, parentDir)

from policy_val_qLearning import *


param_List_SL = [  {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -2, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.04},
                {'alpha': 0.2, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -1, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -2, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.01},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -2, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.01},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.2, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -1, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.02}
            ]

train_List_SL  = [ {'negRewards': False, 'select_action': 'boltzman'},
                {'negRewards': False, 'select_action': 'epsilon_decay'},
                {'negRewards': False, 'select_action': 'noise_method'},
                {'negRewards': True, 'select_action': 'boltzman'},
                {'negRewards': True, 'select_action': 'epsilon_decay'},
                {'negRewards': True, 'select_action': 'noise_method'}
            ]

param_List_BL = [   {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.04},
                    {'alpha': 0.2, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                    {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -2, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.04},
                    {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -2, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                    {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                    {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                    {'alpha': 0.2, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                    {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02}
            ]

train_List_BL  = [ {'negRewards': False, 'select_action': 'boltzman'},
                {'negRewards': False, 'select_action': 'epsilon_decay'},
                {'negRewards': False, 'select_action': 'noise_method'},
                {'negRewards': True, 'select_action': 'boltzman'},
                {'negRewards': True, 'select_action': 'epsilon_decay'},
                {'negRewards': True, 'select_action': 'noise_method'},
                {'negRewards': True, 'select_action': 'epsilon_decay'},
                {'negRewards': False, 'select_action': 'epsilon_decay'}
            ]

titles  = ['Boltzmann, No Neg.', 'Epsilon Decay, no Neg.', 'Noise Method, No Neg.',
           'Boltzmann, Neg. Rewards', 'Epsilon Decay, Neg. Rewards', 'Noise Method, Neg. Rewards']

sNames  = ['Boltzmann_noNeg', 'Eps_noNeg', 'Noise_noNeg',
           'Boltzmann_neg', 'Eps_neg', 'Noise_neg']

sNames_BL  = ['Boltzmann_noNeg_BL', 'Eps_noNeg_BL', 'Noise_noNeg_BL',
           'Boltzmann_neg_BL', 'Eps_neg_BL', 'Noise_neg_BL']

# .78 over 100 for SL
# .99 over 100 for BL

def isSolved(rewards, target:float = 0.78, windowLen:int = 100):
    start   = 0
    lim     = len(rewards)

    for i in range(windowLen, lim):
        score   = np.mean(rewards[start:i])
        start   += 1

        if score >= target:
            break

    return i

enviro      = gym.make('FrozenLake-v0')
enviro.reset()
env_SL         = enviro.env

enviro3     = gym.make('FrozenLake8x8-v0')
enviro3.reset()
env_BL        = enviro3.env

t_till_rewards = []
train_List = train_List_BL
for env in ['small', 'big']:
    if env == 'small':
        env = env_SL
        param_List  = param_List_SL
        target = .78
        tVal    = 100
    else:
        env = env_BL
        param_List  = param_List_BL
        target = .9
        tVal = 250

    tmp     = {}
    modelParams = {'env': env}
    lim         = len(param_List)

    for i in range(8):
        tmpy    = []
        for j in range(5):
            modelParams.update(param_List[i])
            modelParams['Q'] = np.ones((env.observation_space.n, env.action_space.n))*2
            q_learner = Q_learner(**modelParams)
            q_learner.T = tVal
            if i > 5:
                q_learner.epsilon = 0
            rewards = q_learner.train(**train_List[i])
            tmpy.append(isSolved(rewards, target))
        tmp[i]  = tmpy
        print(i)
    t_till_rewards.append(tmp.copy())