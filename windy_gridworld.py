import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0 #possible actions
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord): #essentially to give the coords if agent takes action out pf the grid to stay in the same position
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds): #gives new state and reward given current position and action (delta)
        new_position = np.array(current) + np.array(delta) + np.array([-1, 0]) * winds[tuple(current)]
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == (3, 7) #terminate state
        return [(1.0, new_state, -1.0, is_done)] #( prob ,next_state, reward, if finished)

    def __init__(self):
        self.shape = (7, 10) #grid shape

        nS = np.prod(self.shape) #number of states
        nA = 4 #number of actions

        # Wind strength
        winds = np.zeros(self.shape) #matrix to store wind values
        winds[:,[3,4,5,8]] = 1 #positions of strength 1
        winds[:,[6,7]] = 2 #position of strenght 2

        # Calculate transition probabilities
        P = {} #probabilties p(s',r|s,a)
        for s in range(nS):
            position = np.unravel_index(s, self.shape) #gives i,j indeces of a state
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds) #transition prob for s up, given grid position, delta and wind value matrix
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds) #deterministic, given s,a one posible transition with prob 1
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds) #[-1,0] up, [0,1] right, [1,0]down, [0,-1]left
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)

        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0), self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == (3,7):
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")