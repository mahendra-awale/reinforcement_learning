import numpy as np
from matplotlib import pyplot as plt
import operator


###############################################################
# SIMPLE IMPLEMENTATION OF POLICY ITERATION ALGORITHM
# SEE THE POLICY_ITERATION.png for algorithm
# Problem: 2D-GRID WORLD
# ACTIONS: UP, DOWN, LEFT and RIGHT
# IMMEDIATE REWARD: -1
# IMMEDIATE REWARD AT GOAL STATE: 1
# ENVIRONMENT IS DETERMINISTIC, WHICH MEANS IF I TAKE AN ACTION "UP",
# I GO UP WITH 100% PROBABILITY, THUS AT GIVEN STATE AND ACTION,
# I ONLY NEED TO CONSIDER ONE SUCCESSIVE STATE
###############################################################

class board:

    ###############################################################
    def __init__(self, x=5, y=5, goal_state=(0, 0)):

        # x and y dim
        self._x, self._y = x, y

        # goal state
        self._goal_state = goal_state

        # goal_state reward
        self._goal_state_reward = 1

        # reward everywhere else
        self._immediate_reward = -1

        # define Vs (i.e. value functions for each state)
        self._Vs = np.zeros((x, y))

        # define policy
        self._Pi = self.set_random_uniform_policy()

    ###############################################################
    def set_random_uniform_policy(self):
        ''''
        generate random policy
        '''
        policy = []
        for i in range(len(self._Vs)):
            p = []
            for j in range(len(self._Vs[i])):
                action = np.random.choice(["RIGHT", "LEFT", "UP", "DOWN"])
                p.append(action)
            policy.append(p)
        # set action to "NONE FOR TERMINAL STATE"
        policy[self._goal_state[0]][self._goal_state[1]] = "NONE"

        return policy

    ###############################################################
    def get_action_values(self, state, Vs):

        # I USED DETERMINISTIC ENV, SO PROBABILITY IS 1
        # IN CASE OF PROBABILISTIC ENV, FOR GIVEN ACTION
        # ALL THE POSSIBLE STATES THAT CAN BE REACHED NEEDS TO BE CONSIDERS.
        # i.e. (S, A) = sum (Ps'1 * (reward + lmbd * Vs'1) + Ps'2 * (reward + lmbd * V's2) +...))
        prob = 1

        # DISCOUNT FACTOR, THIS NEEDS TO BE THERE OTHERWISE INFINITE STATE VALUE
        lmbd = 0.9

        # set the appropriate immediate reward
        if state[0] == self._goal_state[0] and state[1] == self._goal_state[1]:
            return {"NONE": prob * (self._goal_state_reward + lmbd * Vs[state[0]][state[1]])}

        reward = self._immediate_reward

        # right action
        if (state[1] + 1) < self._y:
            value_r = prob * (reward + lmbd * Vs[state[0]][state[1] + 1])
        else:
            value_r = prob * (reward + lmbd * Vs[state[0]][state[1]])

        # left action
        if (state[1] - 1) >= 0:
            value_l = prob * (reward + lmbd * Vs[state[0]][state[1] - 1])
        else:
            value_l = prob * (reward + lmbd * Vs[state[0]][state[1]])

        # up action
        if (state[0] - 1) >= 0:
            value_u = prob * (reward + lmbd * Vs[state[0] - 1][state[1]])
        else:
            value_u = prob * (reward + lmbd * Vs[state[0]][state[1]])

        # down action
        if (state[0] + 1) < self._x:
            value_d = prob * (reward + lmbd * Vs[state[0] + 1][state[1]])
        else:
            value_d = prob * (reward + lmbd * Vs[state[0]][state[1]])

        actions = ["RIGHT", "LEFT", "UP", "DOWN"]
        action_values = dict(zip(actions, [value_r, value_l, value_u, value_d]))
        return action_values

    ###############################################################
    def evaulate_policy(self, current_Vs, Pi):
        ''''
        evaluate the current policy
        '''
        new_Vs = np.zeros((self._x, self._y))
        for i in range(len(current_Vs)):
            for j in range(len(current_Vs[i])):
                # policy says take this action
                action = Pi[i][j]

                # get possible action values
                state = (i, j)
                action_values = self.get_action_values(state, current_Vs)

                # value of action suggested by policy
                action_value = action_values[action]

                new_Vs[i][j] = action_value
        return new_Vs

    ###############################################################
    def iterative_policy_improvement(self):

        while True:
            # if policy is stable we will break  while loop
            is_policy_stable = True

            # evaluate the current policy. Remebered that we start with random policy
            Vs = self.evaulate_policy(self._Vs, self._Pi)

            for i in range(self._x):
                for j in range(self._y):

                    # suggested action under current policy
                    curr_action = self._Pi[i][j]

                    # now what actions do we get if we act greedily with
                    # respect to current value function: is the action same?
                    state = (i, j)
                    action_values = self.get_action_values(state, Vs)
                    greedy_action = sorted(action_values.items(), key=operator.itemgetter(1), reverse=True)
                    greedy_action = greedy_action[0][0]

                    # set the action to greedy action with respect to current value function
                    # greedy action guarantee that, policy won't get worst
                    self._Pi[i][j] = greedy_action

                    # if there is mismatch in action selected by greedy policy (i.e. new one)
                    # and the previous policy, it means that Vs is still not correct
                    if greedy_action != curr_action:
                        is_policy_stable = False
            self._Vs = Vs
            if is_policy_stable:
                print(self._Pi)
                print(self._Vs)
                break


b = board()
b.iterative_policy_improvement()
