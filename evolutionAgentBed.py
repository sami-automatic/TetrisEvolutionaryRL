from engineLittle import TetrisEngine
import numpy as np 	
from hedonistic_agent import Agent

env = TetrisEngine(10,20)
num_agents = 16
agents = create_random_agents(num_agents)
rewards = []

def create_random_agents(num_agents):
	agents = []
	for i in range(num_agents):
		random_attributes = construct_random_attributes()
		agent = Agent()
		agent.set_attributes(random_attributes)
		agents.append(agent)
	return agents
		
def construct_random_attributes():
	random_attributes = []
	for i in range(8):
		random_value = np.random.uniform(-10.0, 10.0)
		random_attributes.append(random_value)
	return random_attributes
			
def simulate(times):
	for i in range(times):
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
			environment, reward, is_game_over= environment.step(action)
			reward_of_game += reward
	return reward_of_game


def eliminate_half_of_agents():
	global agents
	remaining_agents = []
	
	for agent in agents:
		rewards.append(play_one_game(agent))

	reward_agent_tuple = zip(rewards,agents)
	reward_agent_tuple.sort()

	for i in range (num_agents / 2):
		remaining_agents[i] = reward_agent_tuple[i][1]
	
	agents = remaining_agents

def cross_over_and_multiply():
	reward_agent_tuple = zip(rewards, agents)
	reward_agent_tuple.sort()

	for i in range(num_agents / 2):
		parent1 = reward_agent_tuple[i][1]
		parent2 = reward_agent_tuple[i+1][1]
		
		attributes_of_parent1 = parent1.get_attributes()
		attributes_of_parent2 = parent2.get_attributes()
		
		shuffeled_attributes = shuffle(attributes_of_parent1, attributes_of_parent2)
		offspring = Agent()
		offspring.set_attributes(shuffeled_attributes)
		agents.append(offspring)

	
	

def shuffle(attribute1, attribute2):
	shuffled = []
	
	for i in range(0,8):
		rand1 = np.random.randint(0,9)
		if rand1 % 2 == 0 :
			shuffled.append(attribute1[i])
		else:
			shuffled.append(attribute2[i])

	return shuffled

simulate(100)
print(agents[0].get_attributes())

