from engineLittle import TetrisEngine
import random
import hedonistic_agent

env = TetrisEngine(10,20)

global agents = []
global rewards = []

def createRandomAgents(numOfAgnets):
	agents = []
	for i in range(numOfAgents):
		random = random.uniform(-10.0, 10.0)
		random1 = random.uniform(-10.0, 10.0)
		random2 = random.uniform(-10.0, 10.0)
		random3 = random.uniform(-10.0, 10.0)
		agents.append(hedonistic_agent(random, random1, random2, random3))
	return agents
		

def playOneFullGame(agent):
	isGameOver = false
	environment = env.clear()
	rewardOfTheGame = 0
	while not isGameOver:
		actions = agent.play(environment) 
		for action in actions:
			environment, reward, isGameOver= environment.step(action)
			rewardOfTheGame += reward
	return rewardOftheGame

def eliminateHalfOfAgents():
	remainingAgents = []
	for agent in agents:
		rewards.append(playOneFullGame(agent))

	rewardAgentTuple = zip(rewards,agents)
	rewardAgentTuple.sort()

	for i in range(numOfAgents/2):
		remainingAgents[i] = rewardAgentTuple[i][1]
	
	agents = remainingAgents

def crossOverAndMultiply():
	offSprings = []
	rewardAgentTuple = zip(rewards, agents)
	rewardAgentTuple.sort()
	for i in range(numOfAgents/2):
		parent1 = rewardAgentTuple[i]
		parent2 = rewardAgentTuple[i+1]
		
		
		offSpring = hedonistic_agent(
		offSrings.append(offSpring)
	


