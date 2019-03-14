from engine_DQN import TetrisEngine
from keras.models import load_model
from dqnkeras import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model2.h5')
env = TetrisEngine(5, 9)
state = env.clear()
done = False
cumulative_reward = 0
n_actions_taken = 0
while not done:
    n_actions_taken +=1
    state_shaped = np.reshape(state, [1, 45])
    action_vals = model.predict(state_shaped)
    action = np.argmax(action_vals[0])
    next_state, reward, done, _ = env.step(action)
    next_state_shaped = np.reshape(next_state, [1, 45])
    cumulative_reward += reward
    state = next_state
    print("Reward:", reward)
    plt.imshow(np.rot90(state, 3))
    plt.show()

print("GAME_OVER")
print("cumulative reward:  ", cumulative_reward)
print("actions taken in an episode: ", n_actions_taken)
