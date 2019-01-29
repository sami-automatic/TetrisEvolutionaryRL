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
        self.gamma = 0.75    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
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
    nb_updates = 9000
    batch_size = 250
    nb_games = 250 # to play without changing cached Q function
    env = TetrisEngine(5, 9)
    state_size = env.height*env.width
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)
    #f = open("test", "w+")


    #training with given
    state = env.clear()
    good_combo_1 = [3,5,6,3,4,6,0,2,3,4]
    good_combo_2 = [3,2,4,6]

    for _ in range(100):
        for act in good_combo_1:
            state_shaped = np.reshape(state, [1, 45])
            next_state, reward, done, _ = env.step(act)
            next_state_shaped = np.reshape(next_state, [1, 45])
            agent.remember(state_shaped, act, reward, next_state_shaped, done)
            state = next_state

        state = env.clear()
        for act in good_combo_2:
            state_shaped = np.reshape(state, [1, 45])
            next_state, reward, done, _ = env.step(act)
            next_state_shaped = np.reshape(next_state, [1, 45])
            agent.remember(state_shaped, act, reward, next_state_shaped, done)
            state = next_state

    agent.replay(len(agent.memory))

    for u in range(nb_updates):
        #the DQN is iterated with the same Q function nb_games times
        for g in range(nb_games):
            state = env.clear()
            done = False
            print("New Game")
            #play one tetris game until game over
            while not done:
                # Decide action
                state_shaped = np.reshape(state, [1, 45])
                action = agent.act(state_shaped)
                print(TetrisEngine.actions[action])
                next_state, reward, done, _ = env.step(action)
                next_state_shaped = np.reshape(next_state, [1, 45])

                agent.remember(state_shaped, action, reward, next_state_shaped, done)
                state = next_state
                print(state,"\n Reward:", reward)
            print("GAME_OVER")
            #f.write("\n updates: ", u, "/", nb_updates," games: ", g, "/", nb_games," reward: ", reward)

        #here the Q function gets updated
        #f.write("\n -----------Q Updated----------- ")
        agent.replay(batch_size)

    print("Here how I PLAY")
    state = env.clear
    done = False
    while not done:
        # Decide action
        state_shaped = np.reshape(state, [1, 45])
        action = agent.act(state_shaped)
        print(TetrisEngine.actions[action])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        print(state, "\n Reward:", reward)
    print("GAME_OVER")

    print("How I PLAY IN SCENARIO")
    '''
    [[1. 0. 0. 0. 1. 1. 1. 1. 0.]
     [0. 0. 0. 0. 0. 1. 1. 1. 1.]
     [0. 0. 0. 0. 1. 1. 1. 1. 0.]
     [0. 0. 0. 0. 0. 1. 1. 1. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    '''
    combo =[3,3,5,5]
    state = env.clear()

    for act in combo:
        next_state, reward, done, _ = env.step(act)
        state = next_state
    print(state)
    while not done:
        # Decide action
        state_shaped = np.reshape(state, [1, 45])
        action = agent.act(state_shaped)
        print(TetrisEngine.actions[action])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        print(state, "\n Reward:", reward)
    print("GAME_OVER")


    #f.close()