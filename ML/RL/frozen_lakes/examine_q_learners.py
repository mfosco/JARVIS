import pandas as pd
from policy_val_qLearning import *
import gym, time
import matplotlib.pyplot as plt

parentDir   = os.path.dirname(os.getpwd())
sys.path.insert(0, parentDir)

from policy_val_qLearning import *

small_lake_file     = 'Project_4/small_lake_with_negsII.xlsx'
big_lake_file       = 'Project_4/big_lake_with_negsII.xlsx'

# need to select max for each step action type for each map

'''
need:   negRewards: False, True
        select_action: epsilon_decay, boltzman, noise_method
'''

searches    = ["{'negRewards': False, 'select_action': 'epsilon_decay'}",
               "{'negRewards': True, 'select_action': 'epsilon_decay'}",
               "{'negRewards': False, 'select_action': 'noise_method'}",
               "{'negRewards': True, 'select_action': 'noise_method'}",
               "{'negRewards': False, 'select_action': 'boltzman'}",
               "{'negRewards': True, 'select_action': 'boltzman'}"
               ]

def get_results(f_name:str, searchThings):
    res = {}
    df = pd.read_excel(f_name)

    for s in searchThings:
        tmpDF   = df.loc[df['params_train'] == s]
        tmp_max = max(tmpDF['avg_reward'])
        small_df= tmpDF.loc[tmpDF['avg_reward'] == tmp_max]
        tmpy    = small_df['params_model']
        tmpy_vals   = [(tmpy[key], tmpDF['avg_reward'][key], tmpDF['std_dev_reward'][key]) for key in tmpy.keys()]
        res[s]  = tmpy_vals

    return res

res_SL  = get_results(small_lake_file, searches)
res_BL  = get_results(big_lake_file  , searches)

'''
Best small lake params
'''

param_List_SL = [  {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -2, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.04},
                {'alpha': 0.2, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -1, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -2, 'gamma': 0.999, 'max_iters': 10000, 'move_cost': -0.01},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.95, 'fall_cost': -2, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.01},
                {'alpha': 0.1, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02},
                {'alpha': 0.05, 'epsilon': 1, 'epsilonDecay': 0.99, 'fall_cost': -3, 'gamma': 0.99, 'max_iters': 10000, 'move_cost': -0.02}
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
            ]

train_List_BL  = [ {'negRewards': False, 'select_action': 'boltzman'},
                {'negRewards': False, 'select_action': 'epsilon_decay'},
                {'negRewards': False, 'select_action': 'noise_method'},
                {'negRewards': True, 'select_action': 'boltzman'},
                {'negRewards': True, 'select_action': 'epsilon_decay'},
                {'negRewards': True, 'select_action': 'noise_method'}
            ]

titles  = ['Boltzmann, No Neg.', 'Epsilon Decay, no Neg.', 'Noise Method, No Neg.',
           'Boltzmann, Neg. Rewards', 'Epsilon Decay, Neg. Rewards', 'Noise Method, Neg. Rewards']

sNames  = ['Boltzmann_noNeg', 'Eps_noNeg', 'Noise_noNeg',
           'Boltzmann_neg', 'Eps_neg', 'Noise_neg']

sNames_BL  = ['Boltzmann_noNeg_BL', 'Eps_noNeg_BL', 'Noise_noNeg_BL',
           'Boltzmann_neg_BL', 'Eps_neg_BL', 'Noise_neg_BL']

enviro      = gym.make('FrozenLake-v0')
enviro.reset()
env_SL         = enviro.env

enviro3     = gym.make('FrozenLake8x8-v0')
enviro3.reset()
env_BL        = enviro3.env
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



def get_graphs(env, param_List, train_List, titles):
    #Q           = np.zeros((env.observation_space.n, env.action_space.n))
    modelParams = {'env': env}
    lim         = len(param_List)
    reward_List = [None]*lim

    episodes    = [x for x in range(1, 10001)]
    for i in range(lim):
        modelParams.update(param_List[i])
        modelParams['Q']    = np.zeros((env.observation_space.n, env.action_space.n))
        q_learner   = Q_learner(**modelParams)
        q_learner.T = 200
        rewards     = q_learner.train(**train_List[i])
        isSolved(rewards, .9)
        #print(isSolved(rewards, .99))
        #print(train_List[i])
        print(i)
        reward_List[i]  = rewards
    return reward_List

r_List  = get_graphs(env_BL, param_List_BL, train_List_BL,[])
r_ListSL  = get_graphs(env_SL, param_List_SL, train_List_SL,[])
for i in range(len(r_List)):
    print(train_List_BL[i])
    print(isSolved(r_List[i], .9))

for i in range(len(r_ListSL)):
    print(train_List_SL[i])
    print(isSolved(r_ListSL[i], .78))
'''
        plt.figure()
        plt.title(titles[i])
        plt.grid()


        endy = 1000
        plt.plot(episodes[-endy:], r_List[i][-endy:], 'o-', color="r",
                 label="Rewards")
        plt.legend(loc="best")
        plt.show()
        plt.savefig(sNames_BL[i])
'''





'''
Best big lake params
'''
Ts = [190, 190, 190, 190, 190, 200]#280, 300, 400, 500, 800, 1000] #40, 60, 80, 200] #120, 150, 180, 200, 220, 250]
for i in range(len(Ts)):
    modelParams.update(param_List[i])
    modelParams['Q'] = np.zeros((env.observation_space.n, env.action_space.n))
    q_learner = Q_learner(**modelParams)
    q_learner.T = Ts[i]
    rewards = q_learner.train(**train_List[i])
    print(isSolved(rewards, .9))

t_till_rewards = []
train_List = train_List_BL
for env in ['small', 'big']: #, 'big']:
    if env == 'small':
        env = env_SL
        param_List  = param_List_SL
        target  = .78
        tVal    = 100
        t_decay = .95
    else:
        env = env_BL
        param_List  = param_List_BL
        target  = .99
        tVal    = 500
        t_decay     = .95


    tmp     = {}
    modelParams = {'env': env}
    lim         = len(param_List)

    for i in range(0,6):
        tmpy    = []
        for j in range(5):
            modelParams.update(param_List[i])
            modelParams['Q'] = np.zeros((env.observation_space.n, env.action_space.n))
            q_learner = Q_learner(**modelParams)
            q_learner.T = tVal
            q_learner.temp_decay = t_decay
            #start   = time.time()
            rewards = q_learner.train(**train_List[i])
            #endy    = time.time()
            tmpy.append(isSolved(rewards, target))
        tmp[i]  = tmpy
        print(i)
    t_till_rewards.append(tmp.copy())
