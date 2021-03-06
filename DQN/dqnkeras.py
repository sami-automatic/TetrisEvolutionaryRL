import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import deque
from keras.models import load_model
import random
import time
# import matplotlib.pyplot as plt
from engine_DQN import TetrisEngine
import tensorflow as tf
import string
import keras as K
from functools import reduce

# make sure soft-placement is off
# tf_config = tf.ConfigProto(allow_soft_placement=False)
# tf_config.gpu_options.allow_growth = True
# s = tf.Session(config=tf_config)
# K.backend.set_session(s)


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # 0.9995
        self.learning_rate = 0.003  # 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # with tf.device('/gpu:0'):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
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
        # to try with iterative memory and random sampling
        minibatch = random.sample(self.memory, batch_size)
        general_state = []
        general_target = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            general_state.append(state)
            general_target.append(target_f)

        general_state = np.reshape(general_state, (-1, 45))
        general_target = np.reshape(general_target, (-1, 7))
        self.model.fit(general_state, general_target,
                       batch_size=batch_size, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class Trainer:
    def __init__(self, env, nb_games=10000):
        self.env = env
        self.nb_games = nb_games

    def train(self):
        # fig = plt.figure()
        # fig.show()
        # ax = fig.add_subplot(111)

        print("Playing with", self.env, flush=True)
        # initialize gym environment and the agent
        batch_size = 64
        state_size = self.env.height * self.env.width
        action_size = len(self.env.actions)
        agent = DQNAgent(state_size, action_size)
        rewards = []

        x = []
        y = []
        for g in range(self.nb_games):
            x.append(g)
            state = self.env.clear()
            done = False
            cumulative_reward = 0
            n_actions_taken = 0

            while not done:
                n_actions_taken += 1
                # Decide action
                state_shaped = np.reshape(state, [1, 45])
                action = agent.act(state_shaped)

                next_state, reward, done, _ = self.env.step(action)
                next_state_shaped = np.reshape(next_state, [1, 45])
                cumulative_reward += reward

                agent.remember(state_shaped, action, reward,
                               next_state_shaped, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            agent.update_target_model()
            rewards.append(cumulative_reward)
            y.append(n_actions_taken)

            # ax.plot(x, y, color='b')
            # fig.canvas.draw()
            time.sleep(0.1)
            # print("cumulative reward: ", '{:.3}'.format(cumulative_reward),
            #       "\tepsilon: ",  '{:.6}'.format(agent.epsilon),
            #       "\tepisode: ", g,
            #       "\tnumber_actions: ", n_actions_taken, flush=True)
            model_name = "model_%s.h5" % (self.env)
            try:
                agent.model.save(model_name)
            except:
                print("COULDN'T SAVE MODEL")
        # plt.show(block = False)
        # plt.savefig('model_%s.png' % (self.env))
        # plt.close('all')
        state = self.env.clear()
        done = False
        cumulative_reward = 0
        n_actions_taken = 0
        while not done:
            n_actions_taken += 1
            state_shaped = np.reshape(state, [1, 45])
            action_vals = agent.model.predict(state_shaped)
            action = np.argmax(action_vals[0])
            next_state, reward, done, _ = self.env.step(action)
            next_state_shaped = np.reshape(next_state, [1, 45])
            cumulative_reward += reward
            state = next_state
        print("survived model n_actions_taken", n_actions_taken, flush=True)
        return n_actions_taken
