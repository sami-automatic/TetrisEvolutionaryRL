#This is the modified engine so that we can take  multiple actions and train DQN
#from __future__ import print_function

import numpy as np
import random

shapes = {
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
}

shape_names = ['Z']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def idle(shape, anchor, board):
    return (shape, anchor)


class TetrisEngine:
    def __init__(self, width, height, board=[]):
        self.width = width
        self.height = height
        self.board = np.asarray(board, dtype=np.float32) if len(board) > 0 else np.zeros(
            shape=(width, height), dtype=np.float)
        
        #make those hard_coded from the best survived hedonistic agent
        self.column_count = 20
        self.row_count = 10
        self.touches_another_block_reward = 182.06908759316752
        self.touches_floor_reward = 197.80362431053217
        self.touches_wall_reward = 11.101274843710573
        self.clear_line_reward = 55.32199495763268
        self.height_multiplier_penalty = -0.11704552783216822
        self.hole_penalty = -0.41247700693381895
        self.blockade_penalty = -0.2123814086602633
        self.bumpiness_penalty = -0.18858682144007557

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict(
            [(j, i) for i, j in self.value_action_map.items()])
            
        self.nb_actions = len(self.actions)

        # for running the engine
        self.time = -1
        self.score = -1
        self.anchor = None
        self.shape = None
        self.n_deaths = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.clear()

    def _choose_shape(self):
        maxm = max(self._shape_counts)
        m = [5 + maxm - x for x in self._shape_counts]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                self._shape_counts[i] += 1
                return shapes[shape_names[i]]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        #x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2

        # ATTENTION: Normally it's in the middle!
        # self.anchor = (self.width / 2, 0)
        self.anchor = (0, 0)

        #self.anchor = (x, 0)
        self.shape = self._choose_shape()

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.score += sum(can_clear)
        self.board = new_board

        return sum(can_clear)

    def valid_action_count(self):
        valid_action_sum = 0

        for _, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    actions = [
        [2], [1, 2], [1, 1, 2], [1, 1, 1, 2], [1, 1, 1, 1, 2],
        [4, 2], [4, 1, 2], [4, 1, 1, 2], [4, 1, 1, 1, 2], [4, 1, 1, 1, 1, 2],
        [4, 4, 2], [4, 4, 1, 2], [4, 4, 1, 1, 2], [4, 4, 1, 1, 1, 2], [4, 4, 1, 1, 1, 1, 2],
        [4, 4, 4, 2], [4, 4, 4, 1, 2], [4, 4, 4, 1, 1, 2], [4, 4, 4, 1, 1, 1, 2], [4, 4, 4, 1, 1, 1, 1, 2],
        [5, 2], [5, 1, 2], [5, 1, 1, 2], [5, 1, 1, 1, 2], [5, 1, 1, 1, 1, 2],
        [5, 5, 2], [5, 5, 1, 2], [5, 5, 1, 1, 2], [5, 5, 1, 1, 1, 2], [5, 5, 1, 1, 1, 1, 2],
        [5, 5, 5, 2], [5, 5, 5, 1, 2], [5, 5, 5, 1, 1, 2], [5, 5, 5, 1, 1, 1, 2], [5, 5, 5, 1, 1, 1, 1, 2],
    ]

    def step(self, actions_code):
        consecutive_actions = self.actions[actions_code]
        print("consecutive_actions", consecutive_actions)
        cleared = 0

        for action in consecutive_actions:
            board, done, cleared_with_one_action = self.small_step(action)
            cleared += cleared_with_one_action
        
        state = np.copy(board)
        full_reward = self.calculate_score(state,cleared)
        print("State\n", state)
        return state, full_reward, done, cleared

    def small_step(self, action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](
            self.shape, self.anchor, self.board)
        # Drop each step
        self.shape, self.anchor = soft_drop(
            self.shape, self.anchor, self.board)

        # Update time and reward
        self.time += 1
        ##reward = self.valid_action_count()
        #reward = 1

        #
        cleared = 0
        done = False
        if self._has_dropped():
            self._set_piece(True)
            cleared_lines = self._clear_lines()
            #reward += 10 * cleared_lines
            cleared = cleared_lines
            if np.any(self.board[:, 0]):
                # self.clear()
                self.n_deaths += 1
                done = True
                ##reward = -10
            else:
                self._new_piece()

        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state, done, cleared

    def clear(self):
        self.time = 0
        self.score = 0
        self._new_piece()
        # self.board = np.zeros_like(self.board)

        return self.board

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i),
                           int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]
                                      ) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        return s


    ###from hedonistic agent
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