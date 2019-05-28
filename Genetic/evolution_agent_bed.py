# from engine import TetrisEngine
# import numpy as np
# from hedonistic_agent import Agent

# NUMBER_OF_AGENTS = 16
# env = TetrisEngine(5, 20)
# agents = []
# rewards = []
# reward_agent_tuple = []


# def create_random_agents():
#     for _ in range(NUMBER_OF_AGENTS):
#         rewards = [np.random.uniform(0.0, 200.0) for _ in range(4)]
#         penalties = [np.random.uniform(-0.1, -0.5) for _ in range(4)]
#         print("rewards", rewards, "penalties", penalties)
#         random_genes = rewards + penalties
#         agent = Agent()
#         agent.set_genes(random_genes)
#         agents.append(agent)

# def simulate(times):
#     for _ in range(times):
#         play_all_agents()
#         eliminate_agents()
#         cross_over_and_multiply()
#     print("last reward_agent_tuple", reward_agent_tuple)
#     env = TetrisEngine(5, 20)
#     reward = play_one_game(agents[0])
#     print("Best agent?", agents[0], "best reward?", reward, "with genes", agents[0].get_genes())
#     print("last reward_agent_tuple", reward_agent_tuple)
#     env = TetrisEngine(5, 20)
#     last_reward = play_one_game(agents[-1])
#     print("Worse agent?", agents[len(agents) - 1], "worse reward?", last_reward, "with genes", agents[-1].get_genes())
#     print("last reward_agent_tuple", reward_agent_tuple)

# def play_all_agents():
#     global rewards
#     for agent in agents:
#         reward = play_one_game(agent)
#         rewards.append(reward)


# def play_one_game(agent):
#     global env
#     is_game_over = False
#     environment = env.clear()
#     reward_of_game = 0
#     count = 0
#     while not is_game_over:
#         count += 1
#         actions, score = agent.play(environment)
#         for action in actions:
#             print("Playing", action, "in environment: \n", environment)
#             environment, _, is_game_over, _ = env.step(action)
#             print("Played", action, "in environment: \n", environment)
#         print("\n\n\n\nactions", actions, "\n\n\n\n\n")
#         reward_of_game += score
#         print("Actions played, reward_of_game:", reward_of_game, "is game over?", is_game_over)
#         if (is_game_over):
#             print("Game is over! Environment\n", environment)
#     print("\n\n\n\n\n\n\n\n\ngame is over, returning.......\n", reward_of_game, "\n\n\n")
#     env = TetrisEngine(5, 20)
#     return reward_of_game


# def eliminate_agents():
#     global agents, rewards, reward_agent_tuple
#     print("rewards", rewards)
#     print("agents", agents)
#     zipped = zip(rewards, agents)
#     reward_agent_list = list(zipped)
#     reward_agent_tuple = reward_agent_list[:]
#     print("before reward_agent_tuple", reward_agent_tuple)
#     reward_agent_tuple = sorted(reward_agent_tuple, key=lambda x: x[0], reverse=True)
#     print("after reward_agent_tuple", reward_agent_tuple)
#     agents = [reward_agent_tuple[i][1] for i in range(NUMBER_OF_AGENTS // 2 + 1)]   # +1 because we need the last agent


# def cross_over_and_multiply():
#     global agents
#     for i in range(len(agents) - 1, 2):
#         top = agents[i]
#         bottom = agents[i+1]
#         lhs = top.get_genes()
#         rhs = bottom.get_genes()
#         shuffled_genes = [lhs[x] if np.random.randint(0, 9) % 2 == 0 else rhs[x] for x in range(8)]
#         exposed_genes = expose_to_mutation(shuffled_genes)
#         print("exposed_genes", exposed_genes)
#         offspring = Agent()
#         offspring.set_genes(exposed_genes)
#         agents.append(offspring)

# def expose_to_mutation(genes):
#     exposed_genes = []
#     for gene in genes:
#             mutation_happened = np.random.randint(0,9) == 9
#             if mutation_happened:
#                    exposed_genes.append(np.random.uniform(-10.0, 10.0))
#             else:
#                   exposed_genes.append(gene)
#     return exposed_genes


# create_random_agents()
# simulate(10)
# # print(agents[0].get_genes())
