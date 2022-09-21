
#
# BF derived reference machine for use with AIQ
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3


import random
import sys

import numpy as np
from gym import Env
from gym import spaces
from gym.utils.env_checker import check_env

from .ReferenceMachine import *

from numpy import zeros, ones, array, linspace
from scipy import stats, floor, sqrt
# from string import replace


class BFGymEnv(Env):
    metadata = {"render_modes": []}



    # create a new BF reference machine, default to a tape with 5 symbols
    def __init__( self, refm, render_mode = None ):

        ###########################################################################################
        #BF
        self.num_obs = refm.getNumObs() # assuming that states = observations
        self.num_actions = refm.getNumActions()
        self.num_symbols = refm.getNumObsSyms()
        self.obs_cells = refm.getNumObsCells()

        self.mid_symbol = int((self.num_symbols-1)/2)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_obs)
        self.reset()
        self.reward_range = (-100.0, 100.0)



    ## Overriding method that return only observation to fulfill requirements of OpenAI Gym
    def reset( self, seed=None , options = None ):
        self.given_observations = {}
        self.given_previous_rewards = {}
        # self.state = np.array([0],[self.mid_symbol*self.obs_cells])
        self.current_step = 0
        return (self.mid_symbol*self.obs_cells)


    def step(self, action):
        observations = self.given_observations[self.current_step]
        reward = self.given_previous_rewards[self.current_step]
        self.current_step += 1
        return observations, reward, True if observations is None else False, ""

    def set_reward_and_observation(self, observation, reward, agent):
        # if  (not isinstance(observation,int)):
        #     print("Vstup není Int: " + str(observation) + "typ: " + str(type(observation)) + "krok: " + str(self.current_step))

        # agent._last_obs = self.given_observations[self.step]
        self.given_observations[self.current_step] = observation
        self.given_previous_rewards[self.current_step] = reward

    def render(self, mode="machine"):
        # print(self.input_tape[self.action_cells])
        # print(self.obs_cells)
        return
