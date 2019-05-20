import numpy as np
from engine_DQN import TetrisEngine
from dqnkeras import Trainer
import random
import time
from operator import itemgetter


class GeneticEngineGenerator:
    n_of_engines = 6
    engines = []
    generations = []

    def generate_random_engines(self, NUMBER_OF_ENGINES=6):
        self.n_of_engines = NUMBER_OF_ENGINES
        print("generate_random_engines")
        for _ in range(NUMBER_OF_ENGINES):
            genes = [
                # clear_line_reward
                np.random.uniform(0.0, 20.0),    # 15.0
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
        print("returning best resulting agent env pair")
        return self.generations[0]

    def play_all_engines(self):
        print("play_all_engines")
        for i in self.generations:
            agent = i[0]
            if (agent == [0, 0, 0]):
                engine = i[1]
                print("engine")
                print(engine)
                mean, variance, std = self.train_with_trainer(engine)
                print("play_all_engines mean, variance, std")
                print(mean, variance, std)
                agent = [mean, variance, std]
                i[0] = agent

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
        top = self.generations[0][1]
        bottom = self.generations[1][1]
        lhs = top.get_genes()
        print("lhs", lhs)
        rhs = bottom.get_genes()
        print("rhs", rhs)
        for i in range(2, self.n_of_engines):
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
        for i, gene in enumerate(genes):
            mutation_happened = np.random.randint(0, 100) < 13
            if mutation_happened:
                print("mutation happened in index: ", i)
                val = [-0.1, -1.0]
                if (i == 0):
                    val = [0.0, 20.0]
                print("uniform between", val)
                exposed_genes.append(np.random.uniform(val[0], val[1]))
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
            end = time.time()
            all_steps.append(n_steps)
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


if __name__ == "__main__":
    generator = GeneticEngineGenerator()
    generator.generate_random_engines(10)
    best_agent_env_pair = generator.simulate(10)
    print("Best performed agent is...", best_agent_env_pair[0])
    print("Best performed env is...", best_agent_env_pair[1])
