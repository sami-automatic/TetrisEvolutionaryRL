#THIS CLASS IS NOT GONNA BE USED IN THE FIRST PART OF THE PROJECT SINCE IT USES KERAS
#WE WILL MOVE FURTHER FOR THE FIRST TERM WITH THE OTHER DQN CLASS

from __future__ import division
import argparse
from engine import TetrisEngine

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

class TetrisProcessor(Processor):
    def process_observation(self, observation):
        #flatten the board to feed it in to DQN
        processed_observation = observation.flatten()
        return processed_observation

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype(int)
        return processed_batch

    def process_reward(self, reward):
        #DEniz we can implement the reward here coming from environment actually without changing the engine class

        return reward

# Get the environment and extract the number of actions.
env = TetrisEngine(10,20)
nb_actions = 20
WINDOW_LENGTH = 4

model = Sequential()
model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
model.add(Dense(32), activation = 'relu')
model.add(Dense(nb_actions, activation = 'linear'))
model.compile(loss='mse',print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = TetrisProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
