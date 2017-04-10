#!/usr/bin/env python

import numpy as np
import operator
from random import sample as sample
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

class Space:
    """Space for an agent to navigate through to find the goal state

    Attributes:
        goal (tuple): the position of the goal with in the space
        height (integer): number of rows in the space
        width (integer): number of columns in the space
        reward (float): reward value for finding the goal
        grid (np array): matrix of possible states populated with zeros except for the goal state
    """
    def __init__(self,height=7,width=11,goal=(5,9),reward=2.5):
        self.goal = goal
        self.height = height
        self.width = width
        self.reward = reward
        self.grid = np.zeros((height,width))
        self.grid[self.goal] = self.reward

class Agent:
    """Learning agent that explores a space to find the optimal path to a goal state

    Attributes:
        directions (dict): compass directions as keys with coordinate adjustment tuples as values
        gamma (float): discount rate for future states
        space (object): space in which the agent explores to find the optimal path to the goal
        move_cost (float): cost for each movement
        move_options (dict): move costs to neighboring states
        trail (list): traceable path the agent followed
    """
    def __init__(self,gamma=.9,space=Space(),state=(0,0),move_cost=0.02,out_path='steps/'):
        self.directions = {'N':(-1,0),'S':(1,0),'E':(0,1),'W':(0,-1)}
        self.out_path = out_path
        self.gamma = gamma
        self.space = space
        self.state = state
        self.step = 0
        self.move_cost = move_cost
        self.move_options = {}
        self.trail = []

    def optimize_path(self,start_state=(0,0)):
        """Repeat until convergence"""
        previous_trail = None
        while self.trail <> previous_trail:
            previous_trail = self.trail
            self.seek(start_state)
            #print(self.space.grid)

    def plot_path(self):
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        ax.xaxis.tick_top()
        ax.annotate('A',xy=(self.state[1],self.state[0]),color='steelblue',horizontalalignment='center',verticalalignment='center')
        res = ax.imshow(self.space.grid,cmap=plt.cm.Greys,interpolation='nearest')
        s = (5 - len(str(self.step))) * '0' + str(self.step)

        plt.savefig(self.out_path + 'step' + s + '.png')

    def seek(self,start_state=(0,0)):
        """Follow the path of least resistance until the goal is found"""
        self.trail = []
        self.state = start_state
        while self.state <> self.space.goal:
            #self.plot_path()
            self.move()
            self.step += 1
        print('->'.join(self.trail))

    def get_expected_value(self,value):
        """Calculate the expected value for moving to a state
        
        Returns:
            float: expected value based on discount and move cost
        """
        return value * self.gamma - self.move_cost

    def get_best_direction(self):
        """Explore neighboring states and return best option

        Returns:
            string: key for the best available direction
        """
        max_v = -float("inf") 
        opts = set()
        p = self.move_options[self.state]

        for k in p:
            if p[k] == max_v:
                opts.add(k)
            if p[k] > max_v:
                max_v = p[k]
                opts = set([k])

        self.space.grid[self.state] = self.get_expected_value(max_v) 
        return sample(opts,1)[0]

    def move(self):
        """Update the position of the agent"""
        self.get_moves()
        direction = self.get_best_direction()
        self.trail.append(direction)
        prev_state = self.state
        self.state = tuple(map(operator.add,self.state,self.directions[direction]))
        ev = {direction: self.get_expected_value(self.space.grid[self.state])}
        self.move_options[prev_state].update(ev)

    def check_state(self,direction):
        """Calculate the expected value of a neighboring state
        
        Args:
            direction (string): key to look up the expected value in a neighboring state

        Returns:
            float: expected value of a movement in the provided direction
        """
        next_state = tuple(map(operator.add,self.state,self.directions[direction]))
        return self.get_expected_value(self.space.grid[next_state])

    def get_moves(self):
        """Populate a dictionary with the moves available from current state"""
        moves = {}
        y,x = self.state
        if y > 0:
            moves['N'] = self.check_state('N')
        if y < self.space.height - 1:
            moves['S'] = self.check_state('S')
        if x > 0:
            moves['W'] = self.check_state('W')
        if x < self.space.width - 1:
            moves['E'] = self.check_state('E')
        self.move_options[self.state] = moves

def main():
    Agent().optimize_path()

if __name__=='__main__':
    main()

