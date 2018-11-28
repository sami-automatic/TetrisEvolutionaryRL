from engineLittle import TetrisEngine
import numpy as np
import importlib
Agent = importlib.import_module(
    'hedonistic_agent', 'TetrisEvolutionaryRL\\hedonistic_agent.py').Agent
NUMBER_OF_AGENTS = 16
env = TetrisEngine(10, 20)
agents = []
rewards = []


def create_random_agents():
    for _ in range(NUMBER_OF_AGENTS):
        random_attributes = construct_random_attributes()
        agent = Agent()
        agent.set_attributes(random_attributes)
        agents.append(agent)


def construct_random_attributes():
    random_attributes = []
    for _ in range(8):
        random_value = np.random.uniform(-10.0, 10.0)
        random_attributes.append(random_value)
    return random_attributes


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
    remaining_agents = []
    for agent in agents:
        rewards.append(play_one_game(agent))
    reward_agent_tuple = zip(rewards, agents)
    sorted(reward_agent_tuple)
    for i in range(NUMBER_OF_AGENTS // 2):
        remaining_agents[i] = list(reward_agent_tuple)[i][1]
    agents = remaining_agents


def cross_over_and_multiply():
    reward_agent_tuple = zip(rewards, agents)
    sorted(reward_agent_tuple)
    for i in range(NUMBER_OF_AGENTS // 2):
        parent1 = list(reward_agent_tuple)[i][1]
        parent2 = list(reward_agent_tuple)[i+1][1]
        attributes_of_parent1 = parent1.get_attributes()
        attributes_of_parent2 = parent2.get_attributes()
        shuffeled_attributes = shuffle(
            attributes_of_parent1, attributes_of_parent2)
        offspring = Agent()
        offspring.set_attributes(shuffeled_attributes)
        agents.append(offspring)


def shuffle(attribute1, attribute2):
    shuffled = []
    for i in range(0, 8):
        rand1 = np.random.randint(0, 9)
        if rand1 % 2 == 0:
            shuffled.append(attribute1[i])
        else:
            shuffled.append(attribute2[i])

    return shuffled


create_random_agents()
simulate(100)
print(agents[0].get_attributes())
