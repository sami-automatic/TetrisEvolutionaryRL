from engine import TetrisEngine
import numpy as np
from hedonistic_agent import Agent

NUMBER_OF_AGENTS = 16
env = TetrisEngine(10, 20)
agents = []
rewards = []
reward_agent_tuple = []


def create_random_agents():
    for _ in range(NUMBER_OF_AGENTS):
        random_attributes = [np.random.uniform(-10.0, 10.0) for _ in range(8)]
        agent = Agent()
        agent.set_attributes(random_attributes)
        agents.append(agent)


def simulate(times):
    for _ in range(times):
        play_all_agents()
        eliminate_agents()
        cross_over_and_multiply()


def play_all_agents():
    global rewards
    for agent in agents:
        reward = play_one_game(agent)
        rewards.append(reward)


def play_one_game(agent):
    global env
    is_game_over = False
    environment = env.clear()
    reward_of_game = 0
    while not is_game_over:
        actions = agent.play(environment)
        for action in actions:
            environment, reward, is_game_over = environment.step(action)
            reward_of_game += reward
    return reward_of_game


def eliminate_agents():
    global agents, rewards, reward_agent_tuple
    zipped = zip(rewards, agents)
    reward_agent_list = list(zipped)
    reward_agent_tuple = reward_agent_list[:]
    sorted(reward_agent_tuple, key=lambda x: x[0])
    agents = [reward_agent_tuple[i][1] for i in range(NUMBER_OF_AGENTS // 2 + 1)]   # +1 because we need the last agent


def cross_over_and_multiply():
    global agents
    for i in range(len(agents) - 1):
        top = agents[i]
        bottom = agents[i+1]
        lhs = top.get_attributes()
        rhs = bottom.get_attributes()
        shuffled_attributes = [lhs[x] if np.random.randint(0, 9) % 2 == 0 else rhs[x] for x in range(8)]
        offspring = Agent()
        offspring.set_attributes(shuffled_attributes)
        agents.append(offspring)


create_random_agents()
simulate(10)
# print(agents[0].get_attributes())
