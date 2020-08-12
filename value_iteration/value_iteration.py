import numpy as np
from matplotlib import pyplot as plt
import operator


###############################################################
# SIMPLE IMPLEMENTATION OF VALUE ITERATION ALGORITHM
# Problem: 2D-GRID WORLD
# ACTIONS: UP, DOWN, LEFT and RIGHT
# IMMEDIATE REWARD: -1
# IMMEDIATE REWARD AT GOAL STATE: 1
# ENVIRONMENT IS DETERMINISTIC, WHICH MEANS IF I TAKE AN ACTION "UP",
# I GO UP WITH 100% PROBABILITY, THUS AT GIVEN STATE AND ACTION,
# I ONLY NEED TO CONSIDER ONE SUCCESSIVE STATE
###############################################################

class board:

    def __init__(self, x=5, y=5, goal_state=(0, 0)):

        # x and y dim
        self._x, self._y = x, y

        # goal state
        self._goal_state = goal_state

        # goal_state reward
        self._goal_state_reward = 1

        # reward everywhere else
        self._immediate_reward = -1

        # state values
        self._values = np.zeros((x, y))

    def get_value_function(self, state):

        # I USED DETERMINISTIC ENV, SO PROBABILITY IS 1
        # IN CASE OF PROBABILISTIC ENV, FOR GIVEN ACTION
        # ALL THE POSSIBLE STATES THAT CAN BE REACHED NEEDS TO BE CONSIDERS.
        # i.e. (S, A) = sum (Ps'1 * (reward + lmbd * Vs'1) + Ps'2 * (reward + lmbd * V's2) +...))
        prob = 1

        # DISCOUNT FACTOR, THIS NEEDS TO BE THERE OTHERWISE INFINITE STATE VALUE
        lmbd = 0.9

        # set the appropriate immediate reward
        if state[0] == self._goal_state[0] and state[1] == self._goal_state[1]:
            return "NONE_ACTION", prob * (self._goal_state_reward + lmbd * self._values[state[0]][state[1]])

        reward = self._immediate_reward

        # right action
        if (state[1] + 1) < self._y:
            value_r = prob * (reward + lmbd * self._values[state[0]][state[1] + 1])
        else:
            value_r = prob * (reward + lmbd * self._values[state[0]][state[1]])

        # left action
        if (state[1] - 1) >= 0:
            value_l = prob * (reward + lmbd * self._values[state[0]][state[1] - 1])
        else:
            value_l = prob * (reward + lmbd * self._values[state[0]][state[1]])

        # up action
        if (state[0] - 1) >= 0:
            value_u = prob * (reward + lmbd * self._values[state[0] - 1][state[1]])
        else:
            value_u = prob * (reward + lmbd * self._values[state[0]][state[1]])

        # down action
        if (state[0] + 1) < self._x:
            value_d = prob * (reward + lmbd * self._values[state[0] + 1][state[1]])
        else:
            value_d = prob * (reward + lmbd * self._values[state[0]][state[1]])

        moves = ["right", "left", "up", "down"]
        d = dict(zip(moves, [value_r, value_l, value_u, value_d]))
        d_sorted = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        best_action = d_sorted[0][0]
        qvalue = d_sorted[0][1]
        return best_action, qvalue

    def get_policy(self):

        policy = []
        for x in range(self._x):
            p = []
            for y in range(self._y):
                state = (x, y)
                best_action, qvalue = self.get_value_function(state)
                p.append(best_action)
            policy.append(p)
        return policy


    def run_value_iteration(self):

        import copy
        iter = 0
        while True:

            iter+=1
            Vs = copy.deepcopy(self._values)
            for x in range(self._x):
                for y in range(self._y):
                    state = (x, y)
                    best_action, qvalue = self.get_value_function(state)
                    Vs[x][y] = qvalue

            if np.abs((np.sum(self._values)) - np.sum(Vs)) < 0.0000001:
                print("converged after %s iterations, lets stop now", iter)
                print("value functions")
                policy = self.get_policy()
                print(policy)
                print(Vs)
                plt.imshow(self._values, cmap="rainbow")
                plt.show()
                break
            self._values = copy.deepcopy(Vs)

b = board()
b.run_value_iteration()

