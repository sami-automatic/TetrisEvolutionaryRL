# manuel DQN

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import deque
from keras.models import load_model
import random
from engine_DQN import TetrisEngine


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995  # 0.9995
        self.learning_rate = 0.0001  # 0.0001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
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
        self.model.fit(general_state, general_target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training of our DQN

if __name__ == "__main__":

    # initialize gym environment and the agent
    batch_size = 64  # 64
    nb_games = 10000
    env = TetrisEngine(5, 9)
    state_size = env.height * env.width
    action_size = len(env.actions)
    agent = DQNAgent(state_size, action_size)

    for g in range(nb_games):
        state = env.clear()
        done = False
        cumulative_reward = 0
        n_actions_taken = 0

        while not done:
            n_actions_taken += 1

            # Decide action
            state_shaped = np.reshape(state, [1, 45])
            action = agent.act(state_shaped)

            next_state, reward, done, _ = env.step(action)
            next_state_shaped = np.reshape(next_state, [1, 45])
            cumulative_reward += reward

            agent.remember(state_shaped, action, reward,
                           next_state_shaped, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        agent.update_target_model()
        print("cumulative reward:\t", '{:.3}'.format(cumulative_reward),
              "\t epsilon:\t",  '{:.6}'.format(agent.epsilon),
              "\t episode:\t", g, "number_actions:\t", n_actions_taken)
        agent.model.save('model3.h5')
