from engine_DQN import TetrisEngine

NUMBER_OF_ENGINES = 16


class GeneticEngineGenerator:
    engines = []
    rewards = []
    reward_engine_tuple = []

    def generate_random_engines():
        for _ in range(NUMBER_OF_ENGINES):
            parameters = [
                # clear_line_reward
                np.random.uniform(0.1, 30.0)    # 15.0
                # height_penalty
                np.random.uniform(-0.1, -1.0)   # -0.810066
                # hole_penalty
                np.random.uniform(-0.1, -0.5)   # -0.36
                # bumpiness_penalty
                np.random.uniform(-0.1, -0.5)   # -0.18
                # game_over_penalty
                np.random.uniform(-0.1, -0.5)   # -0.86
            ]
            engine = TetrisEngine(5, 9, parameters)
            engines.append(engine)

    def simulate(times):
        for _ in range(times):
            play_all_engines()
            eliminate_engines()
            cross_over_and_multiply()

    def play_all_engines():
        for engine in engines:
            reward = train(engine)
            rewards.append(reward)

    def eliminate_engines(self):
        zipped = zip(rewards, engines)
        reward_engine_list = list(zipped)
        reward_engine_tuple = reward_engine_list[:]
        reward_engine_tuple = sorted(
            reward_engine_tuple, key=lambda x: x[0], reverse=True)
        engines = [reward_engine_tuple[i][1]
            for i in range(NUMBER_OF_ENGINES // 2 + 1)]

    def cross_over_and_multiply(self):
        for i in range(len(engines) - 1, 2):
            top = engines[i]
            bottom = engines[i + 1]
            lhs = top.get_genes()
            rhs = bottom.get_genes()
            shuffled_genes = [lhs[x] if np.random.randint(
                0, 9) % 2 == 0 else rhs[x] for x in range(8)]
            exposed_genes = expose_to_mutation(shuffled_genes)
            offspring = TetrisEngine(5, 9, exposed_genes)
            engines.append(offspring)

    def expose_to_mutation(self, genes):
        exposed_genes = []
        for gene in genes:
            mutation_happened = np.random.randint(0, 9) == 9
               if mutation_happened:
                    exposed_genes.append(np.random.uniform(-10.0, 10.0))
                else:
                    exposed_genes.append(gene)
        return exposed_genes

    def train(self, env):
        model = load_model('model2.h5')
        state = env.clear()
        done = False
        cumulative_reward = 0
        n_actions_taken = 0
        while not done:
            n_actions_taken += 1
            state_shaped = np.reshape(state, [1, 45])
            action_vals = model.predict(state_shaped)
            action = np.argmax(action_vals[0])
            next_state, reward, done, _ = env.step(action)
            next_state_shaped = np.reshape(next_state, [1, 45])
            cumulative_reward += reward
            state = next_state
            plt.imshow(np.rot90(state, 3))
            plt.show()
        return cumulative_reward


if __name__ == "__main__":
    generator = GeneticEngineGenerator()
    generator.generate_random_engines()
    generator.simulate(10)
    print("Best engine genes", generator.engines[0].get_genes())
