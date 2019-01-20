#manuel DQN

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import deque
import random
from engine_DQN import TetrisEngine

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(34, input_dim=self.state_size, activation='relu'))
        model.add(Dense(34, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        #to try with iterative memory and random sampling
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#Training of our DQN

if __name__ == "__main__":
    # initialize gym environment and the agent
    nb_updates = 250
    batch_size = 1000
    nb_games = 1000 # to play without changing cached Q function
    env = TetrisEngine(5,10)
    state_size = env.height*env.width
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)
    f = open("test", "w+")

    for u in range(nb_updates):
        #the DQN is iterated with the same Q function nb_games times
        for g in range(nb_games):
            state = env.clear()
            done = False
            #play tetris until game over
            while not done:
                # Decide action
                state_shaped = np.reshape(state, [1, 50])
                action = agent.act(state_shaped)
                next_state, reward, done, _ = env.step(action)
                next_state_shaped = np.reshape(next_state, [1, 50])

                agent.remember(state_shaped, action, reward, next_state_shaped, done)
                state = next_state
            f.write("\n updates: ", u, "/", nb_updates," games: ", g, "/", nb_games," reward: ", reward)

        #here the Q function gets updated
        f.write("\n -----------Q Updated----------- ")
        agent.replay(batch_size)

    f.close()