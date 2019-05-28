# import os
# import numpy as np
# from engine_DQN import TetrisEngine
# from dqnkeras import Trainer
# import random
# import time
# from operator import itemgetter
# import multiprocessing


# class GeneticEngineGenerator:
#     n_of_engines = 6
#     engines = []
#     generations = []

#     def generate_random_engines(self, NUMBER_OF_ENGINES=6):
#         self.n_of_engines = NUMBER_OF_ENGINES
#         print("generate_random_engines", flush=True)
#         for _ in range(NUMBER_OF_ENGINES):
#             genes = [
#                 # clear_line_reward
#                 np.random.uniform(0.0, 20.0),    # 15.0
#                 # height_penalty
#                 np.random.uniform(-0.1, -1.0),   # -0.810066
#                 # hole_penalty
#                 np.random.uniform(-0.1, -1.0),   # -0.36
#                 # bumpiness_penalty
#                 np.random.uniform(-0.1, -1.0),   # -0.18
#                 # game_over_penalty
#                 np.random.uniform(-0.1, -1.0)    # -0.86
#             ]
#             print("Creating engine with genes...", genes, flush=True)
#             engine = TetrisEngine(5, 9, genes)
#             # self.engines.append(engine)
#             self.generations.append([[0, 0, 0], engine])

#     def simulate(self, cycle):
#         for i in range(cycle):
#             print("simulate: ", i, flush=True)
#             self.play_all_engines()
#             print("generation: ", i, flush=True)
#             for i in self.generations:
#                 print("agent: ", i[0], "env: ", i[1], flush=True)
#             self.eliminate_engines()
#             self.cross_over_and_multiply()
#         print("returning best resulting agent env pair", flush=True)
#         return self.generations[0]


#     def worker(self, engine, i, queue):
#         print("worker process started for: ", engine, "in index: ", i, flush=True)
#         # INFO: change these variables in terms of determine times and games
#         mean, variance, std = self.train_with_trainer(engine, 10, 10)
#         print("worker process finished:  mean, variance, std", flush=True)
#         print(mean, variance, std, flush=True)
#         agent = [mean, variance, std]
#         queue.put([i, agent])


#     def play_all_engines(self):
#         print("play_all_engines", self.generations, flush=True)
#         queue = multiprocessing.Queue()
#         jobs = []
#         for i, val in enumerate(self.generations):
#             print(self.generations)
#             agent = val[0]
#             if (agent == [0, 0, 0]):
#                 engine = val[1]
#                 print("engine", flush=True)
#                 print(engine, flush=True)
#                 p = multiprocessing.Process(target=self.worker, args=(engine, i, queue))
#                 jobs.append(p)
#                 p.start()
#         for job in jobs:
#             print("worker job joined.", flush=True)
#             i, val = queue.get()
#             print("queue i", i, " val", val)
#             self.generations[i][0] = val
#             job.join()
#         # print("play_all_engines finished with jobs", jobs, "generation", self.generations, flush=True)

#     def eliminate_engines(self):
#         print("eliminate_engines", flush=True)

#         for i in self.generations:
#             print("agent is", flush=True)
#             print(i[0], flush=True)
#             print("", flush=True)

#         for i in self.generations:
#             print("==========", flush=True)
#             print("GENERATION", flush=True)
#             print("agent is", flush=True)
#             print(i[0], flush=True)
#             print("printing agent's engine", flush=True)
#             print(i[1], flush=True)
#             print("==========", flush=True)
#         self.last_generation = self.generations[-1]
#         print("last_generation agent", flush=True)
#         print(self.last_generation[0], flush=True)
#         print("last_generation env", flush=True)
#         print(self.last_generation[1], flush=True)
#         # np.sort(self.last_generation, axis=0)  # sort according to mean
#         self.generations.sort(key=lambda x: x[0][0], reverse=True)
#         del self.generations[2:]  # keep the first two kill others
#         print("after deletion generations", flush=True)
#         print(self.generations, flush=True)
#         for i in self.generations:
#             print("==========", flush=True)
#             print("GENERATION", flush=True)
#             print("agent is", flush=True)
#             print(i[0], flush=True)
#             print("printing agent's engine", flush=True)
#             print(i[1], flush=True)
#             print("==========", flush=True)

#     def cross_over_and_multiply(self):
#         print("cross_over_and_multiply", flush=True)
#         print("old engines", flush=True)
#         print("==============", flush=True)
#         for i in self.generations:
#             eng = i[1]
#             print("printing engine: ", flush=True)
#             print(eng, flush=True)
#         print("==============", flush=True)
#         ln = len(self.generations)
#         print("generations length: ", flush=True)
#         print(ln, flush=True)
#         top = self.generations[0][1]
#         bottom = self.generations[1][1]
#         lhs = top.get_genes()
#         print("lhs", lhs, flush=True)
#         rhs = bottom.get_genes()
#         print("rhs", rhs, flush=True)
#         for i in range(2, self.n_of_engines):
#             shuffled_genes = [lhs[x] if np.random.randint(
#                 0, 9) % 2 == 0 else rhs[x] for x in range(5)]
#             print("shuffled_genes", flush=True)
#             print(shuffled_genes, flush=True)
#             exposed_genes = self.expose_to_mutation(shuffled_genes)
#             print("exposed_genes", flush=True)
#             print(exposed_genes, flush=True)
#             offspring = TetrisEngine(5, 9, exposed_genes)
#             print("offspring", flush=True)
#             print(offspring, flush=True)
#             self.generations.append([[0, 0, 0], offspring])
#             print("self.generations after appending offspring", flush=True)
#             print("==============", flush=True)
#             for i in self.generations:
#                 agent = i[0]
#                 print("agent", agent, flush=True)
#                 eng = i[1]
#                 print("printing engine: ", flush=True)
#                 print(eng, flush=True)
#             print("==============", flush=True)

#         print("cross_over_and_multiply finished", flush=True)
#         print("new engines", flush=True)
#         print("==============", flush=True)
#         for i in self.generations:
#             eng = i[1]
#             print("printing engine: ", flush=True)
#             print(eng, flush=True)
#         print("==============", flush=True)

#     def expose_to_mutation(self, genes):
#         print("expose_to_mutation", flush=True)
#         exposed_genes = []
#         for i, gene in enumerate(genes):
#             mutation_happened = np.random.randint(0, 100) < 13
#             if mutation_happened:
#                 print("mutation happened in index: ", i, flush=True)
#                 val = [-0.1, -1.0]
#                 if (i == 0):
#                     val = [0.0, 20.0]
#                 print("uniform between", val, flush=True)
#                 exposed_genes.append(np.random.uniform(val[0], val[1]))
#             else:
#                 exposed_genes.append(gene)
#         return exposed_genes

#     def train_with_trainer(self, env, times=5, num_games=400):
#         print("train_with_trainer", flush=True)
#         trainer = Trainer(env, num_games)
#         all_steps = []

#         for i in range(times):
#             print("=========================", flush=True)
#             print("step: ", i, flush=True)
#             start = time.time()
#             print("start", start, flush=True)
#             n_steps = trainer.train()
#             print("n_steps", n_steps, flush=True)
#             all_steps.append(n_steps)
#             print("all_steps", all_steps, flush=True)
#             end = time.time()
#             print("Training finished in....", end - start, flush=True)
#             print("=========================", flush=True)

#         print("calculating mean variance and std from this..", flush=True)
#         print(all_steps, flush=True)
#         print("while my engine is...", env)
#         mean = np.mean(all_steps)
#         variance = np.var(all_steps)
#         std = np.std(all_steps)
#         print("mean", mean, flush=True)
#         print("variance", variance, flush=True)
#         print("std", std, flush=True)
#         return mean, variance, std


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     generator = GeneticEngineGenerator()
#     # INFO: change this variable to set random of engines in each generation
#     generator.generate_random_engines(5)
#     # INFO: change this variable to set generation
#     best_agent_env_pair = generator.simulate(10)
#     print("Best performed agent is...", best_agent_env_pair[0], flush=True)
#     print("Best performed env is...", best_agent_env_pair[1], flush=True)
