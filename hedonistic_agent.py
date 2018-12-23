from engine import TetrisEngine
from math import floor
import numpy as np
import operator


class Agent(object):
    def __init__(self,
                 column_count=20,
                 row_count=5,
                 touches_another_block_reward=6.0,
                 touches_floor_reward=10.0,
                 touches_wall_reward=5.0,
                 clear_line_reward=150.0,
                 height_multiplier_penalty=-0.03,
                 hole_penalty=-0.01,
                 blockade_penalty=-0.3,
                 bumpiness_penalty=-0.2):

        self.column_count = column_count
        self.row_count = row_count
        self.touches_another_block_reward = touches_another_block_reward
        self.touches_floor_reward = touches_floor_reward
        self.touches_wall_reward = touches_wall_reward
        self.clear_line_reward = clear_line_reward
        self.height_multiplier_penalty = height_multiplier_penalty
        self.hole_penalty = hole_penalty
        self.blockade_penalty = blockade_penalty
        self.bumpiness_penalty = bumpiness_penalty
        self.mid = floor(row_count / 2)  # middle of the rows.

    def __str__(self):
        return  "\nAGENT: %s, %s, %s, %s, %s, %s, %s, %s \n" % (self.touches_another_block_reward, self.touches_floor_reward, self.touches_wall_reward, self.clear_line_reward, self.height_multiplier_penalty, self.hole_penalty, self.blockade_penalty, self.bumpiness_penalty) 

    def __repr__(self):
        return  "\nAGENT: %s, %s, %s, %s, %s, %s, %s, %s \n" % (self.touches_another_block_reward, self.touches_floor_reward, self.touches_wall_reward, self.clear_line_reward, self.height_multiplier_penalty, self.hole_penalty, self.blockade_penalty, self.bumpiness_penalty)         

    # 0: left,
    # 1: right,
    # 2: hard_drop,
    # 3: soft_drop,
    # 4: rotate_left,
    # 5: rotate_right,
    # 6: idle

    # Possibilities:
    # Can hard-drop to any location without any rotation. ((2), (1, 2), (1, 1, 2), ...)
    # Can rotate left or right in any given action list. ((4, 2), (4, 1, 2), (4, 1, 1, 2), ...)
    # TODO: Play all possible combinations and get the biggest scored one.
    # 0 -- ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 1 -- ( 4 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 2 -- ( 4 ), ( 4 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 3 -- ( 4 ), ( 4 ), ( 4 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 1 -- ( 5 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 2 -- ( 5 ), ( 5 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # 3 -- ( 5 ), ( 5 ), ( 5 ), ( 2 ), ( 1, 2 ), ( 1, 1, 2 ), ( 1, 1, 1, 2 ), ( 1, 1, 1, 1, 2 )
    # Gets state, returns action list
    actions = [
        [2], [1, 2], [1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 1, 2],
        [4, 2], [4, 1, 2], [4, 1, 1, 2], [4, 1, 1, 1, 2], [4, 1, 1, 1, 1, 2],
        [4, 4, 2], [4, 4, 1, 2], [4, 4, 1, 1, 2], [4, 4, 1, 1, 1, 2], [4, 4, 1, 1, 1, 1, 2],
        [4, 4, 4, 2], [4, 4, 4, 1, 2], [4, 4, 4, 1, 1, 2], [4, 4, 4, 1, 1, 1, 2], [4, 4, 4, 1, 1, 1, 1, 2],
        [5, 2], [5, 1, 2], [5, 1, 1, 2], [5, 1, 1, 1, 2], [5, 1, 1, 1, 1, 2],
        [5, 5, 2], [5, 5, 1, 2], [5, 5, 1, 1, 2], [5, 5, 1, 1, 1, 2], [5, 5, 1, 1, 1, 1, 2],
        [5, 5, 5, 2], [5, 5, 5, 1, 2], [5, 5, 5, 1, 1, 2], [5, 5, 5, 1, 1, 1, 2], [5, 5, 5, 1, 1, 1, 1, 2],
    ]

    def play(self, state):
        # action list can be 0, 0, 0, 2  (left, left, left, hard_drop)
        action_list = []
        env = TetrisEngine(self.row_count, self.column_count, state)
        obs = env.board
        print("Initial state\n", obs)
        reward_action_pair = {}
        number = 4
        count = -1
        is_game_over = False
        for execute in range(8):
            if (execute is not 4):
                print("execute:", execute % 4, number)
            else:
                number = 5
                continue
            for i in range(5):
                for _ in range(execute % 4):
                    obs, _, is_game_over, cleared = env.step(number)
                    print("action: ", number, "\nSTATE:\n", obs)
                    # print(str(number) + ",", end=" ")
                for _ in range(i):
                    obs, _, is_game_over, cleared = env.step(1)
                    print("action: ", 1, "\nSTATE:\n", obs)
                    # print("1,", end=" ")
                obs, _, is_game_over, cleared = env.step(2)
                print("action: ", 2, "\nSTATE:\n", obs)
                # print(2, end=" ")
                count = count + 1
                print("")
                print("count: " + str(count), "execute: " + str(execute), "number: " + str(number))
                score = self.calculate_score(obs, cleared)
                reward_action_pair[count] = score
                env = TetrisEngine(self.row_count, self.column_count, state)
                obs = state
        print("reward_action_pair", reward_action_pair)
        sorted_by_value = sorted(reward_action_pair.items(), key=lambda x: x[1])[::-1]
        print("sorted_by_value", sorted_by_value, len(sorted_by_value), len(self.actions))
        max_value = sorted_by_value[0][1]
        print("max_value", max_value)
        max_value_array = [sorted_by_value[0][0]]
        for i in sorted_by_value[1:]:
            if (i[1] == max_value):
                max_value_array.append(i[0])
        print("Max value array", max_value_array)
        actual_actions = []
        for i in max_value_array:
            actual_actions.append(self.actions[i])
        print("Actual actions", actual_actions)
        actual_actions = sorted(actual_actions, key=len)
        print("Actual actions sorted", actual_actions)
        print("\n\nFound! Returning..") 
        print(actual_actions[0])
        print("With total reward: ", max_value) 
        return actual_actions[0], max_value, is_game_over

    def get_genes(self):
        genes = [
            self.touches_another_block_reward,
            self.touches_floor_reward,
            self.touches_wall_reward,
            self.clear_line_reward,
            self.height_multiplier_penalty,
            self.hole_penalty,
            self.blockade_penalty,
            self.bumpiness_penalty
        ]
        return genes

    def set_genes(self, genes):
        self.touches_another_block_reward = genes[0]
        self.touches_floor_reward = genes[1]
        self.touches_wall_reward = genes[2]
        self.clear_line_reward = genes[3]
        self.height_multiplier_penalty = genes[4]
        self.hole_penalty = genes[5]
        self.blockade_penalty = genes[6]
        self.bumpiness_penalty = genes[7]

    def calculate_edge(self, obs, i, j):
        point = 0.0     # cumulative point of edge
        # neighbours of cell.
        neighbours = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]
        got_floor_point = False
        for (x, y) in neighbours:
            if j == self.column_count - 1 and not got_floor_point:   # touching the floor
                point += self.touches_floor_reward
                got_floor_point = True
            try:                # has neighboring index
                if obs[x][y]:   # touches another block
                    point += self.touches_another_block_reward
            except IndexError:                                  # point is on the edge
                if y == self.column_count and got_floor_point:  # if we added the floor point, don't add extra 2.5!
                    continue
                point += self.touches_wall_reward
        return point

    def calculate_hole_score(self, hole_count):
        return self.hole_penalty * hole_count

    def get_highest_index(self, list):
        index = 0
        for (i, value) in enumerate(list):
            if value == 1 and i > index:
                index = i
        return index

    def calculate_bumpiness_score(self, obs):
        penalty = 0.0
        for i in range(self.row_count - 1):
            diff = abs(self.get_highest_index(
                obs[i]) - self.get_highest_index(obs[i + 1]))
            penalty += diff * self.bumpiness_penalty
        return penalty

    def is_starting_cell(self, i, j):
        # (i == mid or i == (mid - 1))
        return j == 0 and i == 0

    def calculate_blockaded_score(self, blockaded_count):
        return self.blockade_penalty * blockaded_count

    def calculate_cleared_score(self, cleared):
        return self.clear_line_reward * cleared

    def calculate_score(self, obs, cleared):
        edge_score = 0.0
        hole_count = 0
        blockaded_count = 0
        for (i, row) in enumerate(obs):
            for (j, cell) in enumerate(row):
                # starting position of block. ignore!
                if self.is_starting_cell(i, j):
                    continue
                if cell:
                    edge_point = self.calculate_edge(obs, i, j)
                    edge_score += edge_point
                else:
                    try:
                        if obs[i - 1][j]:   # empty and top of blocked
                            blockaded_count += 1
                        else:               # top is not blocked, normal hole
                            hole_count += 1
                    except IndexError:      # top doesn't exist, normal hole
                        hole_count += 1
        total_score, hole_score, bumpiness_score, blockaded_score = (0, 0, 0, 0)
        cleared_score = self.calculate_cleared_score(cleared)
        if cleared == 0:
            hole_score = self.calculate_hole_score(hole_count)
            bumpiness_score = self.calculate_bumpiness_score(obs)
            blockaded_score = self.calculate_blockaded_score(blockaded_count)
        print("cleared score:", cleared_score)
        total_score = edge_score + hole_score + \
            bumpiness_score + blockaded_score + cleared_score
        total_score = round(total_score, 2)
        print("total score:", total_score, "\n\n")
        return total_score


# Example usage
# state = np.array([[0.]*20 for i in range(5)])
# state = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
#          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]]
# hedonistic_agent = Agent()  # Agent(True) for debug output.
# hedonistic_agent.play(state)
