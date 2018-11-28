from engineLittle import TetrisEngine
from math import floor
import numpy as np

# gets state, returns action list


class Agent(object):
    def __init__(self,
                 debug=False,
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

        self.debug = debug
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

    def play(self, state):
        action_list = []
        env = TetrisEngine(self.row_count, self.column_count, state)
        obs = env.board
        # done = false
        # while done is false:
        #     obs, reward, done, cleared = env.step(2)
        #     debug and print("obs: \n", obs, "\nreward:",reward, "\ndone:", done, "\n\n")
        #     calculate_score(obs, cleared)
        for i in range(5):
            for _ in range(i):
                self.debug and print("go right")
                obs, reward, done, cleared = env.step(1)    # right
                self.debug and print(
                    "obs: \n", obs, "\nreward:", reward, "\ndone:", done, "\n\n")
            self.debug and print("go down")
            obs, reward, done, cleared = env.step(2)        # down
            print("obs: \n", obs, "\ndone:", done)
            self.calculate_score(obs, cleared)
        print("\n============\nfinal state\n============\nobs: \n",
              obs, "\nreward:", reward, "\ndone:", done, "\n\n")
        return action_list

    def get_attributes(self):
        attributes = [
            self.touches_another_block_reward,
            self.touches_floor_reward,
            self.touches_wall_reward,
            self.clear_line_reward,
            self.height_multiplier_penalty,
            self.hole_penalty,
            self.blockade_penalty,
            self.bumpiness_penalty
        ]
        return attributes

    def set_attributes(self, attributes):
        self.touches_another_block_reward = attributes[0]
        self.touches_floor_reward = attributes[1]
        self.touches_wall_reward = attributes[2]
        self.clear_line_reward = attributes[3]
        self.height_multiplier_penalty = attributes[4]
        self.hole_penalty = attributes[5]
        self.blockade_penalty = attributes[6]
        self.bumpiness_penalty = attributes[7]

    def calculate_edge(self, obs, i, j):
        point = 0.0     # cumulative point of edge
        neighbours = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]        # neighbours of cell.
        got_floor_point = False
        for (x, y) in neighbours:
            if j == self.column_count - 1 and not got_floor_point:   # touching the floor
                self.debug and print("touching the floor, 5 point!")
                point += self.touches_floor_reward
                got_floor_point = True
            try:                # has neighboring index
                if obs[x][y]:   # touches another block
                    self.debug and print(
                        "touching another block, 3 point! touches: (%d, %d)" % (x, y))
                    point += self.touches_another_block_reward
            except IndexError:                                  # point is on the edge
                if y == self.column_count and got_floor_point:  # if we added the floor point, don't add extra 2.5!
                    self.debug and print("floor point already added.")
                    continue
                self.debug and print(
                    "is on edge, 2.5 point! cant find: (%d, %d)" % (x, y))
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
                    self.debug and print("starting position, skipping..")
                    continue
                if cell:
                    self.debug and print("i:", i, "\tj:", j, "\tvalue:", cell)
                    edge_point = self.calculate_edge(obs, i, j)
                    self.debug and print("point:", edge_point)
                    edge_score += edge_point
                else:
                    try:
                        if obs[i - 1][j]:   # empty and top of blocked
                            blockaded_count += 1
                        else:               # top is not blocked, normal hole
                            hole_count += 1
                    except IndexError:      # top doesn't exist, normal hole
                        hole_count += 1
        self.debug and print("score:", edge_score)
        self.debug and print("empty row count:", hole_count)
        total_score, hole_score, bumpiness_score, blockaded_score = (
            0, 0, 0, 0)
        cleared_score = self.calculate_cleared_score(cleared)
        if cleared == 0:
            hole_score = self.calculate_hole_score(hole_count)
            bumpiness_score = self.calculate_bumpiness_score(obs)
            blockaded_score = self.calculate_blockaded_score(blockaded_count)
        self.debug and print("empty cell score:", hole_score)
        self.debug and print("bumpiness score:", bumpiness_score)
        self.debug and print("blockaded score:", blockaded_score)
        print("cleared score:", cleared_score)
        total_score = edge_score + hole_score + \
            bumpiness_score + blockaded_score + cleared_score
        total_score = round(total_score, 2)
        print("total score:", total_score, "\n\n")


# Example usage
state = np.array([[0.]*20 for i in range(5)])
hedonistic_agent = Agent()  # Agent(True) for debug output.
hedonistic_agent.play(state)
