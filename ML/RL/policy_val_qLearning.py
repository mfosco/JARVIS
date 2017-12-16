import gym
import numpy as np
import math
import operator



class Policy_value_iter(object):

    def __init__(self, env, gamma, max_iters, tol):
        '''
        Parameters
        ----------
        env: gym.core.Environment
          The environment to compute value iteration for. Must have nS,
          nA, and P as attributes.
        gamma: float
          Discount factor, must be in range [0, 1)
        policy: np.array
          The policy to evaluate. Maps states to actions.
        max_iters: int
          The maximum number of iterations to run before stopping.
        tol: float
          Determines when value function has converged.
        '''
        self.env    = env
        self.gamma  = gamma
        self.max_iters  = int(max_iters)
        self.tol    = tol

    def evaluate_policy(self, policy):
        '''
        Evaluate the value of a policy.
        See page pg 123 pdf of the Sutton and Barto Second Edition
        book.
        http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf
        Parameters
        ----------
        policy: np.array
          The policy to evaluate. Maps states to actions.
        Returns
        -------
        np.ndarray
          The value for the given policy
        '''
        gamma   = self.gamma
        tol     = self.tol
        value_func_old = np.random.rand(self.env.nS)
        value_func_new = np.zeros(self.env.nS)
        for iter in range(self.max_iters):
            delta = 0
            for s in range(self.env.nS):
                vs = 0
                actions = [policy[s]]
                for a in actions:
                    for possible_next_state in self.env.P[s][a]:
                        prob_action = possible_next_state[0]  # probability of action always in 0th spot
                        cur_reward = possible_next_state[2]  # current reward in slot 2
                        future_reward = gamma * value_func_old[possible_next_state[1]]
                        vs += prob_action * (cur_reward + future_reward)
                diff = abs(value_func_old[s] - vs)
                delta = max(delta, diff)
                value_func_new[s] = vs

            if delta <= tol:
                break
            value_func_old = value_func_new
        return value_func_new, iter

    def value_function_to_policy(self, value_function):
        '''
        Output action numbers for each state in value_function.
        Parameters
        ----------
        value_function: np.ndarray
            Value of each state.
        Returns
        -------
        np.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
        '''
        policy = np.zeros(self.env.nS, dtype='int')
        for s in range(self.env.nS):
            maxvsa = float('-inf')
            maxa = float('-inf')
            for a in range(self.env.nA):
                vsa = 0
                for possible_next_state in self.env.P[s][a]:
                    prob_action = possible_next_state[0]
                    cur_reward = possible_next_state[2]
                    future_reward = self.gamma * value_function[possible_next_state[1]]
                    vsa += prob_action * (cur_reward + future_reward)
                if vsa > maxvsa:
                    maxvsa = vsa
                    maxa = a
            policy[s] = maxa
        return policy

    def improve_policy(self, value_func, policy):
        '''Given a policy and value function improve the policy.
        See page  pg 127 pdf of the Sutton and Barto Second Edition
        book.
        http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf
            Parameters
        ----------
        value_func: np.ndarray
          Value function for the given policy.
        policy: dict or np.array
          The policy to improve. Maps states to actions.
        Returns
        -------
        bool, np.ndarray
          Returns true if policy changed. Also returns the new policy.
        '''
        gamma   = self.gamma
        stable = True
        for s in range(self.env.nS):
            old_action = policy[s]
            maxvsa = float('-inf')
            maxa = float('-inf')
            for a in range(self.env.nA):
                vsa = 0
                for possible_next_state in self.env.P[s][a]:
                    '''prob_action     = possible_next_state[0]
                    cur_reward      = possible_next_state[2]
                    future_reward   = gamma * value_func[possible_next_state[1]]'''
                    prob_action, future_reward, cur_reward = possible_next_state[0:3]
                    future_reward = gamma * value_func[future_reward]
                    vsa += prob_action * (cur_reward + future_reward)
                if vsa > maxvsa:
                    maxvsa = vsa
                    maxa = a
            if maxa != old_action: stable = False
            policy[s] = maxa
        return stable, policy

    def policy_iteration(self):
        '''Runs policy iteration.
        See page pg 123 pdf of the Sutton and Barto Second Edition
        book.
            http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf
        You should use the improve_policy and evaluate_policy methods to
        implement this method.
        Returns
        -------
        (np.ndarray, np.ndarray, int, int)
           Returns optimal policy, value function, number of policy
           improvement iterations, and number of value iterations.
        '''
        policy = np.zeros(self.env.nS, dtype='int')
        value_func = np.zeros(self.env.nS)
        stable = False
        iters = 0
        eval_iters = 0
        while not stable:
            value_func, iter = self.evaluate_policy(policy)
            eval_iters += iter
            stable, policy = self.improve_policy(value_func, policy)
            iters += 1
        return policy, value_func, iters, eval_iters

    def value_iteration(self):
        '''Runs value iteration for a given gamma and environment.
        Page 127 of of the Barto and Sutton pdf
            http://ufal.mff.cuni.cz/~straka/courses/npfl114/2016/sutton-bookdraft2016sep.pdf
        Parameters
        ----------
        Returns
        -------
        np.ndarray, iteration
          The value function and the number of iterations it took to converge.
        '''
        tol     = self.tol
        gamma   = self.gamma
        value_func_old = np.random.rand(self.env.nS)
        value_func_new = np.zeros(self.env.nS)
        for iter in range(self.max_iters):
            delta = 0.0
            for s in range(self.env.nS):
                maxvsa = float('-inf')
                for a in range(self.env.nA):
                    vsa = 0
                    for possible_next_state in self.env.P[s][a]:
                        prob_action = possible_next_state[0]
                        cur_reward = possible_next_state[2]
                        if not value_func_new[possible_next_state[1]]:
                            future_reward = gamma * value_func_old[possible_next_state[1]]
                        else:
                            future_reward = gamma * value_func_new[possible_next_state[1]]
                        vsa += prob_action * (cur_reward + future_reward)
                    if vsa > maxvsa:
                        maxvsa = vsa
                diff = abs(value_func_old[s] - maxvsa)
                delta = max(delta, diff)
                value_func_new[s] = maxvsa
            if delta <= tol: break
            value_func_old = value_func_new
        return value_func_new, iter

    def get_policy_str(self, policy, action_names):
        '''Print the policy in human-readable format.
        Parameters
        ----------
        policy: np.ndarray
          Array of state to action number mappings
        action_names: dict
          Mapping of action numbers to characters representing the action.
        '''
        policy_str = policy.astype('str')
        for action_num, action_name in action_names.items():
            np.place(policy_str, policy == action_num, action_name)
        return policy_str

class Q_learner(object):

    def __init__(self, env, Q, epsilon: float = 1.0, epsilonDecay: float = .99,
                 alpha: float = .1, gamma: float = .99,
                 max_iters: int = int(1e4),
                 move_cost: float = -.02, fall_cost: float = -.5,
                 temperature: float = 1e10, temp_decay: float = .95):
        self.epsilon    = epsilon
        self.epsilonDecay   = epsilonDecay
        self.env        = env
        self.alpha      = alpha
        self.gamma      = gamma
        self.Q          = Q
        self.max_iters  = max_iters
        self.move_cost  = move_cost
        self.fall_cost  = fall_cost
        self.T          = temperature
        self.temp_decay = temp_decay
        self.noTemp     = False


    def learn_ep(self, selectionMethod, negReward = False, episode: int = 0, render = False):
        '''
        :param selectionMethod: method for how the learner picks new actions
        :param negReward: whether there are negative rewards or not
        :param episode: the episode number
        :param render: if the result should be rendered
        :return: the total reward from this learning episode
        '''
        curr_state      = self.env.reset()
        t_reward        = 0
        max_iters       = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        alpha           = self.alpha
        gamma           = self.gamma

        for i in range(max_iters):
            if render:
                self.env.render()

            action = selectionMethod(curr_state, episode)

            observation, reward, done, info     = self.env.step(action)
            t_reward    += reward

            if negReward and done and reward == 0:
                reward  = self.fall_cost
            elif negReward and not done:
                reward  = self.move_cost

            self.Q[curr_state, action] += alpha * (reward + gamma *
                                                   np.max(self.Q[observation, :]) - self.Q[curr_state, action])
            curr_state  = observation
            if done: break
        return t_reward

    def boltzmann_strategy(self, curr_state, episode):
        '''
        :param curr_state: current state
        :param episode: unused
        :return: the selected action
        '''
        tau = self.T
        n   = self.env.action_space.n
        probabilities = np.zeros(n)

        for i in range(n):
            nom = math.exp(self.Q[curr_state, i] / tau)
            denom = sum(math.exp(val / tau) for val in self.Q[curr_state, :])

            probabilities[i] = nom / denom

        if self.noTemp:
            index, value = max(enumerate(probabilities), key=operator.itemgetter(1))
            probs   = [0]*n
            probs[index]    = 1
            probabilities = probs

        action = np.random.choice(range(n), p=probabilities)

        self.T *= self.temp_decay
        if self.T < .01:
            self.T = .09
            self.noTemp  = True
        return action

    def episode_noise_method(self, curr_state, episode):
        '''
        :param curr_state:  the current state
        :param episode:     the episode number
        :return:            selected action
        '''
        action = np.argmax(self.Q[curr_state, :] + np.random.randn(1, self.env.action_space.n) * (1. / (episode + 1)))
        return action

    def epsilon_decay_method(self, curr_state, episode):
        '''
        :param curr_state:      the current state (an integer to go to the correct spot in the Q matrix)
        :param episode:         unused.
        :return:                selected action
        '''
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.Q[curr_state, :])
        else:
            action = self.env.action_space.sample()

        self.epsilon *= self.epsilonDecay
        return action

    def train(self, select_action = 'epsilon_decay', negRewards: bool = False) -> list:
        '''
        :param select_action:   which selection method to pick from. Possible options are:
                                epsilon_decay, noise_method or boltzmann
        :param negRewards:      whether negative rewards are being used or not
        :return:                a list of rewards for each episode
        '''
        reward_per_ep   = [None]*self.max_iters
        origEpsilon     = self.epsilon
        origT           = self.T

        choices = {'epsilon_decay': self.epsilon_decay_method,
                    'noise_method': self.episode_noise_method,
                    'boltzmann':     self.boltzmann_strategy}

        action_selector     = choices.get(select_action, None)
        if action_selector == None:
            raise ("Invalid action selector method")

        for i in range(self.max_iters):

            reward_per_ep[i]    = self.learn_ep(action_selector, negRewards, i)

        self.epsilon    = origEpsilon
        self.T          = origT
        return reward_per_ep


