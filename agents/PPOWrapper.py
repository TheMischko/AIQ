#
# Q(lambda) algorithm for Sutton and Barto page 184
#
# Copyright Shane Legg 2011
# Released under GNU GPLv3
#
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv

from CustomCallback import CustomCallback
from refmachines.BfGymEnv import BFGymEnv
from .Agent import Agent
from gym import spaces
from numpy import zeros, ones
import numpy as np
from random import randint, randrange, random
import sys, gym
import refmachines
from refmachines.BFGym import BFGym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class PPOWrapper(Agent):

    def __init__( self, refm, disc_rate):

        Agent.__init__( self, refm, disc_rate )

        self.refm = refm
        self.num_states  = refm.getNumObs() # assuming that states = observations
        self.obs_symbols = refm.getNumObsSyms()
        self.obs_cells   = refm.getNumObsCells()

        self.agent = None

        #self.Lambda  = Lambda

        #self.env = BFGym()
        self.reset()
        # self.rollout_buffer = RolloutBuffer(buffer_size=3, observation_space=spaces.Discrete(self.refm.getNumObs()),
        #                                     action_space=spaces.Discrete(self.refm.getNumActions()))


    def reset( self ):
        if self.agent != None:
            del self.agent
        self.env = BFGymEnv(self.refm)
        # check_env(self.env)
        self.state  = 0
        self.action = 0

        self.agent = PPO("MlpPolicy", self.env, verbose=0)
        self.rollout_buffer = RolloutBuffer(buffer_size=3, observation_space=spaces.Discrete(self.refm.getNumObs()),
                                            action_space=spaces.Discrete(self.refm.getNumActions()))
        # self.agent.learn(total_timesteps=25000)
        #, actor_critic='MLPActorCritic', ac_kwargs=dict(hidden_sizes=[16,16])


    def __str__( self ):
        return "PPO("  + str(self.state) + "," \
               + str(self.action) + ")"


    def perceive( self, observations, reward ):

        if isinstance(observations, int):
            observations = [observations]

        if len(observations) != self.obs_cells:
            raise NameError("VPG recieved wrong number of observations!")

        self.env.set_reward_and_observation(observations, reward, self.agent)
        # convert observations into a single number for the new state
        # nstate = 0
        if self.agent._last_obs is None:
            self.agent._last_obs = self.env.reset()  # pytype: disable=annotation-type-mismatch
        # self.agent.train()
        for i in range(self.obs_cells):
            nstate = observations[i] * self.obs_symbols**i
            self.action, self.state = self.agent.predict(nstate)
            self.agent.set_env(self.env,False)
            # self.agent =  self.agent.learn(total_timesteps=1)
            # env = VecEnv(num_envs= 1,observation_space=nstate,action_space=self.env.getNumActions)
            callback = CustomCallback()
            self.rollout_buffer = self.agent.collect_rollouts(env=self.env,rollout_buffer = self.rollout_buffer,n_rollout_steps=1,callback= callback)
            # print("Rolloout Collected")
        #
            self.agent.train()
        #     self.agent.learn(total_timesteps=1)
        return self.action

