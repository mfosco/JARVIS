import gym, time
import os
#import matplotlib as plt
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
import pylab as pl
import csv

parentDir   = os.path.dirname(os.getpwd())
sys.path.insert(0, parentDir)

from policy_val_qLearning import *

def evalPolicy(env, policy, num_iters:int = 1000):
    res     = [0]*num_iters

    for i in range(num_iters):
        env.reset()
        s, r, d, _  = env.step(policy[env.s])
        total_reward= r

        while d is False:
            s, r, d, _  = env.step(policy[s])
            total_reward += r
        res[i]  = total_reward
    return res


def write_results_to_file(file_name, d):
    header = [x for x in d[0].keys()] # header of eval criteria
    fin = format_data(header, d)
    fin.insert(0, header)
    try:
        with open(file_name, "w") as file_out:
            writer = csv.writer(file_out)
            for f in fin:
                writer.writerow(f)
            file_out.close()
    except:
        print('writing failed')
    return

def format_data(header, d):
    len_d = len(d)
    formatted = [[]] * len_d
    len_header = len(header)

    indxForm = 0
    indx = 0
    for x in d:
        tmp = [None] * len_header
        for j in header:
            tmp[indx] = x[j]
            indx += 1
        indx = 0
        formatted[indxForm] = tmp
        indxForm += 1
    return(formatted)

def make_param_grid(d):
    des_dict    = {des_key: d[des_key] for des_key in d.keys()}
    the_grid    = ParameterGrid(des_dict)
    return(the_grid)

def make_q_models():
    models      = {'epsilon': [1], 'epsilonDecay': [.95, .99],
            'alpha': [.05, .1, .2],  # 2,5, 10, 50, 10000],
             'gamma' : [.99, .999], 'max_iters': [int(1e4)],
             'move_cost': [-.01, -.02, -.04, -.08],
             'fall_cost': [-.5, -1, -2, -3]}

    return [models]

def make_train_combos():
    res = []
    d   = make_train_params()
    gridy   = make_param_grid(d)

    for x in gridy:
        res.append(x)
    return(res)


def make_train_params():
    params      = {'negRewards': [False, True],
                   'select_action':   ['epsilon_decay', 'noise_method', 'boltzmann']
                }
    return params

def get_Q_stats(rewards):
    res = {}
    res['avg_reward']   = np.mean(rewards[-1000:])
    res['std_dev_reward'] = np.std(rewards[-1000:])
    return(res)

def fit_training(tup):
    modelParams, trainList, m, j  = tup

    q_orig  = modelParams['Q'].copy()
    res     = [None]*len(trainList)
    i = 0

    for p in trainList:
        q_learner = Q_learner(**modelParams)
        rewards = q_learner.train(**p)
        stats = get_Q_stats(rewards)
        stats['params_model'] = m
        stats['params_train'] = p
        res[i] = stats
        modelParams['Q']    = q_orig.copy()
        i += 1
    print(j)
    return(res)

def test_q_models(env):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    trainList       = make_train_combos()
    modelD          = make_q_models()
    grid_models     = make_param_grid(modelD[0])
    models          = [x for x in grid_models]
    limModels       = len(models)
    limT            = len(trainList)
    #res             = [None]*limT*limModels
    modelParams     = {'Q': Q, 'env': env}
    i               = 0
    pool            = mp.Pool(processes=6)
    zeModels        = [None]*limModels
    print('Total: {}'.format(limT*limModels))
    for i in range(limModels):
        m                = models[i].copy()
        models[i]['Q']   = Q.copy()
        models[i]['env'] = env
        #modelParams.update(m)
        #tmp             = modelParams.copy()
        zeModels[i]     = (models[i], trainList, m, i)
        i += 1

    res = []
    for i in range(limModels):
        res += fit_training(zeModels[i])
    res = pool.map(fit_training, zeModels)

    return(res)




'''
env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

for i_episode in range(20):
    obs     = env.reset()
    for t in range(100):
        env.render()
        print(obs)
        action      = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
'''
enviro      = gym.make('FrozenLake-v0')
enviro.reset()
env_SL         = enviro.env

# policy, value iteration SL
pvi_smallLake   = Policy_value_iter(env_SL, .9, 1e3, 1e-3)
start   = time.time()
policy_SL, value_polEval_SL, iters_pol_SL, eval_iters_pol_SL   = pvi_smallLake.policy_iteration()
end     = time.time()
print(end - start)
start   = time.time()
value_func_val_iter_SL, iters_val_SL                = pvi_smallLake.value_iteration()
end     = time.time()
print(end - start)

numAvg  = 100
thousRewardsPol_SL  = [0]*numAvg
for j in range(numAvg):
    thousRewardsPol_SL[j]   = evalPolicy(env_SL, policy_SL)
    if j % 10 == 0:
        print(j)

polFromVal  = pvi_smallLake.value_function_to_policy(value_func_val_iter_SL)

thousRewardsVal_SL     = [0]*numAvg
for j in range(numAvg):
    thousRewardsVal_SL[j]   = evalPolicy(env_SL, polFromVal)
    if j % 10 == 0:
        print(j)

print('Mean reward through policy iteration: {}'.format(np.mean(thousRewardsPol_SL)))
print('Standard deviation, policy iteration: {}'.format(np.std(thousRewardsPol_SL)))
print('Mean reward through value  iteration: {}'.format(np.mean(thousRewardsVal_SL)))
print('Standard deviation, value  iteration: {}'.format(np.std(thousRewardsVal_SL)))


# Qlearner SL
# Test Params
sName       = 'small_lake_with_negs.csv'
test_params_SL   = test_q_models(env_SL)
z = [item for sublist in test_params_SL for item in sublist]

write_results_to_file(sName, z)

'''
Q = np.zeros((env_SL.observation_space.n, env_SL.action_space.n))
sl_neg_decay      = Q_learner(env_SL, Q, alpha = .1, gamma=.99, max_iters = int(1e4),
                        move_cost = -.04, fall_cost=-2)
rewards_sl_neg_decay = sl_neg_decay.train('epsilon_decay', True)
print(sum(rewards_sl_neg_decay[-1000:]))

sl_neg_noise      = Q_learner(env_SL, Q, alpha = .1, gamma=.99, max_iters = int(1e4),
                        move_cost = -.04, fall_cost=-2)
rewards_sl_neg_noise = sl_neg_noise.train('noise_method', True)
print(sum(rewards_sl_neg_noise[-1000:]))

sl_neg_boltzman      = Q_learner(env_SL, Q, alpha = .1, gamma=.99, max_iters = int(1e4),
                        move_cost = -.04, fall_cost=-2)
rewards_sl_neg_boltzman = sl_neg_boltzman.train('boltzman', True)
print(sum(rewards_sl_neg_boltzman[-1000:]))
'''

# final selections
Q               = np.zeros((env_SL.observation_space.n, env_SL.action_space.n))
q_learner_SL    = Q_learner(env_SL, Q, alpha=.8, gamma=.95, max_iters=200000)
rewards_SL      = q_learner_SL.train()
rewards_SL_noise= q_learner_SL.train(True)
print('Some way: ' + str(sum(rewards_SL)))
print('Noise way: ' + str(sum(rewards_SL_noise)))


######################################################################
# BL
enviro3     = gym.make('FrozenLake8x8-v0')
enviro3.reset()
env_BL        = enviro3.env

# policy, value iter BL
pvi_bigLake     = Policy_value_iter(env_BL, .99, int(1e3), 1e-3)
start   = time.time()
policy_BL, value_polEval_BL, iters_BL, eval_iters_BL = pvi_bigLake.policy_iteration()
end     = time.time()
print(end - start)
start   = time.time()
value_func_val_iter_BL, iters_val_BL    = pvi_bigLake.value_iteration()
end     = time.time()
print(end - start)


numAvg  = 100
thousRewardsPol_BL  = [0]*numAvg
for j in range(numAvg):
    thousRewardsPol_BL[j]   = evalPolicy(env_BL, policy_BL)
    if j % 10 == 0:
        print(j)

polFromVal_BL  = pvi_bigLake.value_function_to_policy(value_func_val_iter_BL)

thousRewardsVal_BL     = [0]*numAvg
for j in range(numAvg):
    thousRewardsVal_BL[j]   = evalPolicy(env_BL, polFromVal_BL)
    if j % 10 == 0:
        print(j)

print('Mean reward through policy iteration: {}'.format(np.mean(thousRewardsPol_BL)))
print('Standard deviation, policy iteration: {}'.format(np.std(thousRewardsPol_BL)))
print('Mean reward through value  iteration: {}'.format(np.mean(thousRewardsVal_BL)))
print('Standard deviation, value  iteration: {}'.format(np.std(thousRewardsVal_BL)))


# Qlearner BL

# Test params
sName       = 'big_lake.csv'
test_params_BL   = test_q_models(env_BL)
z = [item for sublist in test_params_BL for item in sublist]

'''
lim = len(z)
z_reformat = []
for i in range(lim):
    tmp = z[i]
    for key in z[i]:
        tmp[key] = str(tmp[key])
    z_reformat.append(tmp)
'''
write_results_to_file(sName, z)


Q               = np.zeros((env_BL.observation_space.n, env_BL.action_space.n))
q_learner_BL    = Q_learner(env_BL, Q)
rewards_BL      = q_learner_BL.train()
rewards_BL      = q_learner_BL.train(True)
rewards_BLII    = q_learner_BL.train(True)

q_learner_BL_Big = Q_learner(env_BL, Q, max_iters=int(1e4), alpha = .1, epsilonDecay=.999)
rewards_large_BL = q_learner_BL_Big.train(False, int(1e3))

#################################################
# Testing a bunch
sName       = 'small_lake_with_negs.csv'
test_params_SL   = test_q_models(env_SL)
z = [item for sublist in test_params_SL for item in sublist]

write_results_to_file(sName, z)

print('Finished Small lake')

sName       = 'big_lake_with_negs.csv'
test_params_BL   = test_q_models(env_BL)
z = [item for sublist in test_params_BL for item in sublist]

write_results_to_file(sName, z)
