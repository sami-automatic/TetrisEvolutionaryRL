import numpy as np
from engine_DQN import TetrisEngine
from dqnkeras import Trainer
import random
import time
from operator import itemgetter


class GeneticEngineGenerator:
    engines = []
    generations = []

    def generate_random_engines(self, NUMBER_OF_ENGINES=6):
        print("generate_random_engines")
        for _ in range(NUMBER_OF_ENGINES):
            genes = [
                # clear_line_reward
                np.random.uniform(0.0, 15.0),    # 15.0
                # height_penalty
                np.random.uniform(-0.1, -1.0),   # -0.810066
                # hole_penalty
                np.random.uniform(-0.1, -1.0),   # -0.36
                # bumpiness_penalty
                np.random.uniform(-0.1, -1.0),   # -0.18
                # game_over_penalty
                np.random.uniform(-0.1, -1.0)    # -0.86
            ]
            print("Creating engine with genes...", genes)
            engine = TetrisEngine(5, 9, genes)
            # self.engines.append(engine)
            self.generations.append([[0, 0, 0], engine])

    def simulate(self, cycle):
        for i in range(cycle):
            print("simulate: ", i)
            self.play_all_engines()
            self.eliminate_engines()
            self.cross_over_and_multiply()

    def play_all_engines(self):
        print("play_all_engines")
        # generation = []
        for i in self.generations:
            agent = i[0]
            if (agent == [0, 0, 0]):
                engine = i[1]
                print("engine")
                print(engine)
                mean, variance, std = self.train_with_trainer(engine, 2, 5)
                print("play_all_engines mean, variance, std")
                print(mean, variance, std)
                agent = [mean, variance, std]
                i[0] = agent
                # generation.append((agent, engine))
        # print("Printing generation......")
        # print(generation)
        # self.generations.append(generation)

    def eliminate_engines(self):
        print("eliminate_engines")

        for i in self.generations:
            print("agent is")
            print(i[0])
            print("")

        for i in self.generations:
            print("==========")
            print("GENERATION")
            print("agent is")
            print(i[0])
            print("printing agent's engine")
            print(i[1])
            print("==========")
        self.last_generation = self.generations[-1]
        print("last_generation agent")
        print(self.last_generation[0])
        print("last_generation env")
        print(self.last_generation[1])
        # np.sort(self.last_generation, axis=0)  # sort according to mean
        self.generations.sort(key=lambda x: x[0][0], reverse=True)
        del self.generations[2:]  # keep the first two kill others
        print("after deletion generations")
        print(self.generations)
        for i in self.generations:
            print("==========")
            print("GENERATION")
            print("agent is")
            print(i[0])
            print("printing agent's engine")
            print(i[1])
            print("==========")

    def cross_over_and_multiply(self):
        print("cross_over_and_multiply")
        # print("old engines")
        # print("==============")
        # for i in self.engines:
        #     print("printing engine: ")
        #     print(i)
        # print("==============")
        # print("for range in len(self.engines) - 1")
        # print(len(self.engines) - 1)
        # for i in range(len(self.engines) - 1, 2):
        #     top = self.engines[i]
        #     bottom = self.engines[i + 1]
        #     lhs = top.get_genes()
        #     rhs = bottom.get_genes()
        #     shuffled_genes = [lhs[x] if np.random.randint(
        #         0, 9) % 2 == 0 else rhs[x] for x in range(8)]
        #     print("shuffled_genes")
        #     print(shuffled_genes)
        #     exposed_genes = self.expose_to_mutation(shuffled_genes)
        #     print("exposed_genes")
        #     print(exposed_genes)
        #     offspring = TetrisEngine(5, 9, exposed_genes)
        #     print("offspring")
        #     print(offspring)
        #     self.engines.append(offspring)
        #     print("self.engines after appending offspring")
        #     print("==============")
        #     for i in self.engines:
        #         print("printing engine: ")
        #         print(i)
        #     print("==============")
        print("old engines")
        print("==============")
        for i in self.generations:
            eng = i[1]
            print("printing engine: ")
            print(eng)
        print("==============")
        ln = len(self.generations)
        print("generations length: ")
        print(ln)
        for i in range(ln):
            top = self.generations[i][1]
            print("top", top)
            bottom = self.generations[i + 1][1]
            print("bottom", bottom)
            lhs = top.get_genes()
            print("lhs", lhs)
            rhs = bottom.get_genes()
            print("rhs", rhs)
            shuffled_genes = [lhs[x] if np.random.randint(
                0, 9) % 2 == 0 else rhs[x] for x in range(5)]
            print("shuffled_genes")
            print(shuffled_genes)
            exposed_genes = self.expose_to_mutation(shuffled_genes)
            print("exposed_genes")
            print(exposed_genes)
            offspring = TetrisEngine(5, 9, exposed_genes)
            print("offspring")
            print(offspring)
            self.generations.append([[0, 0, 0], offspring])
            print("self.generations after appending offspring")
            print("==============")
            for i in self.generations:
                agent = i[0]
                print("agent", agent)
                eng = i[1]
                print("printing engine: ")
                print(eng)
            print("==============")

        print("cross_over_and_multiply finished")
        print("new engines")
        print("==============")
        for i in self.generations:
            eng = i[1]
            print("printing engine: ")
            print(eng)
        print("==============")

    def expose_to_mutation(self, genes):
        print("expose_to_mutation")
        exposed_genes = []
        for gene in genes:
            mutation_happened = np.random.randint(0, 9) == 9
            if mutation_happened:
                exposed_genes.append(np.random.uniform(-10.0, 10.0))
            else:
                exposed_genes.append(gene)
        return exposed_genes

    def train_with_trainer(self, env, times=5, num_games=400):
        print("train_with_trainer")
        trainer = Trainer(env, num_games)
        all_steps = []

        for i in range(times):
            start = time.time()
            n_steps = trainer.train()
            print("N_steps")
            print(n_steps)
            end = time.time()
            all_steps.append(n_steps)
            print("all_steps")
            print(all_steps)
            print("Printing time in round")
            print(i)
            print("While my environment is")
            print(env)
            print(end - start)

        print("calculating mean variance and std from this..")
        print(all_steps)
        mean = np.mean(all_steps)
        variance = np.var(all_steps)
        std = np.std(all_steps)
        print("mean", mean)
        print("variance", variance)
        print("std", std)
        return mean, variance, std


generator = GeneticEngineGenerator()
generator.generate_random_engines()
generator.simulate(10)
# generator.play_with("Best")
# generator.play_with("Worst")
