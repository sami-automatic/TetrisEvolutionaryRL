#My DQN implementation for tetris using keras
import random
import numpy as np 
from engine import TetrisEngine
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN_agent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size #will be the width of the board (times)*4(number of rotation)
        self.state_size = state_size#size of the board
        self.memory = deque(maxlen=90000)#could be changed with respect to learning samples needed
        self.gamma = 0.0 #discount rate, could be changed
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.01#no idea what it is
        self.epsilon_decay = 0.995#no idea what it is
        self.learning_rate = 0.001#to figure out through experimenting
        self.model = self._build_model()

    def _build_model(self):
        #neural net for dql model
        #2 hidden layers with 32 neurons,same architecture as in the stanford study
        #output neurons as much as actions
        model = Sequential()
        model.add(Dense(32, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(32), activation = 'relu')
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
            optimizer=Adam(lr=self.learning_rate))
        return model

    
    def remember(self,  state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    #epsilon policy acting
    def act(self, state): 
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




    