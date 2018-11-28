from engine import TetrisEngine
import numpy as np
from hedonistic_agent import Agent

NUMBER_OF_AGENTS = 16
env = TetrisEngine(10, 20)
agents = []
rewards = []
reward_agent_tuple = []
reward_agent_list = []


def create_random_agents():
    for _ in range(NUMBER_OF_AGENTS):
        random_attributes = [np.random.uniform(-10.0, 10.0) for _ in range(8)]
        agent = Agent()
        agent.set_attributes(random_attributes)
        agents.append(agent)


def simulate(times):
    for _ in range(times):
        play_all_agents()
        eliminate_half_of_agents()
        cross_over_and_multiply()


def play_all_agents():
    for agent in agents:
        reward = play_one_game(agent)
        rewards.append(reward)


def play_one_game(agent):
    is_game_over = False
    environment = env.clear()
    reward_of_game = 0
    while not is_game_over:
        actions = agent.play(environment)
        for action in actions:
            environment, reward, is_game_over = environment.step(action)
            reward_of_game += reward
    return reward_of_game


def eliminate_half_of_agents():
    global agents
    reward_agent_tuple = zip(rewards, agents)
    sorted(reward_agent_tuple, key=lambda x: x[0])
    reward_agent_list = list(reward_agent_tuple)
    agents = [reward_agent_list[i][1] for i in range(NUMBER_OF_AGENTS // 2)]


def cross_over_and_multiply():
    for i in range(NUMBER_OF_AGENTS // 2):
        top = reward_agent_list[i][1]
        bottom = reward_agent_list[i+1][1]
        lhs = top.get_attributes()
        rhs = bottom.get_attributes()
        shuffled_attributes = [lhs[x] if np.random.randint(0, 9) % 2 == 0 else rhs[x] for x in range(8)]
        offspring = Agent()
        offspring.set_attributes(shuffled_attributes)
        agents.append(offspring)


create_random_agents()
simulate(100)
print(agents[0].get_attributes())
