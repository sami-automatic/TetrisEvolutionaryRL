import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import deque
from keras.models import load_model
import random
import time
import matplotlib.pyplot as plt

from engine_DQN import TetrisEngine
from dqnkeras import Trainer

num_agents = 6
trial_time = 5
survived_steps = []

#beginner genes testing for clear line reward
#clear_line // height // hole // bumpiness // game_over

gene1 = [3.0, -0.05, -0.3, -0.1, -0.5]
gene2 = [1.0, -0.05, -0.3, -0.1, -0.5]
gene3 = [0.5, -0.05, -0.3, -0.1, -0.5]
gene4 = [5.0, -0.05, -0.3, -0.1, -0.5]
gene5 = [7.0, -0.05, -0.3, -0.1, -0.5]
gene6 = [0.7, -0.05, -0.3, -0.1, -0.5]

genes = [gene1,gene2,gene3,gene4,gene5,gene6]

for i in range(num_agents):
    steps = []
    for t in range(trial_time):
        env = TetrisEngine(5,9,genes[i])
        trainer = Trainer(env, 400)
        step = trainer.train()
        steps.append(step)
    survived_steps.append(steps)

print(survived_steps)

'''
num_of_generation = 2
gene1 = [3.0, -0.05, -0.3, -0.1, -0.5]

def create_agents(num_of_agents = 6):
    agents = []
    for a in range(num_of_agents):
        agents.append(env = TetrisEngine(5,9,gene1))
    return agents

#for g in range(num_of_generation):
'''