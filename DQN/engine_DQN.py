# This is the modified engine so that we can take  multiple actions and train DQN
# from __future__ import print_function

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
    def __init__(self, width, height, genes=[15.0, -0.810066, -0.36, -0.18, -0.86], board=[]):
        print("TetrisEngine genes", genes)
        self.width = width
        self.height = height
        self.board = np.asarray(board, dtype=np.float32) if len(board) > 0 else np.zeros(
            shape=(width, height), dtype=np.float)
        self.clear_line_reward = genes[0]
        self.height_penalty = genes[1]
        self.hole_penalty = genes[2]
        self.bumpiness_penalty = genes[3]
        self.game_over_penalty = genes[4]

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle  # suspect
        }
        self.action_value_map = dict(
            [(j, i) for i, j in self.value_action_map.items()])

        self.nb_actions = len(self.actions)

        # for running the engine
        self.score = -1
        self.anchor = None
        self.shape = None
        self.n_deaths = 0

        # used for generating shapes
        self._shape_counts = [0] * len(shapes)

        # clear after initializing
        self.clear()

    def __str__(self):
        return "ENGINE_%s_%s_%s_%s_%s" % (self.clear_line_reward,
                                             self.height_penalty,
                                             self.hole_penalty,
                                             self.bumpiness_penalty,
                                             self.game_over_penalty,
                                             )

    def get_genes(self):
        return [
            self.clear_line_reward,
            self.height_penalty,
            self.hole_penalty,
            self.bumpiness_penalty,
            self.game_over_penalty
        ]

    def set_genes(self, genes):
        self.clear_line_reward = genes[0]
        self.height_penalty = genes[1]
        self.hole_penalty = genes[2]
        self.bumpiness_penalty = genes[3]
        self.game_over_penalty = genes[4]

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
        # x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2

        # ATTENTION: Normally it's in the middle!
        # self.anchor = (self.width / 2, 0)
        self.anchor = (0, 0)

        # self.anchor = (x, 0)
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
        [1, 2], [1, 1, 2], [1, 1, 1, 2], [5, 2], [
            5, 1, 2], [5, 1, 1, 2], [5, 1, 1, 1, 2]
    ]

    def step(self, actions_code):
        consecutive_actions = self.actions[actions_code]
        cleared = 0

        for action in consecutive_actions:
            board, cleared_with_one_action, done = self.one_step(action)
            cleared += cleared_with_one_action
            if done:
                break

        state = np.copy(board)
        full_reward = self.calculate_score(state, cleared, done)
        return state, full_reward, done, cleared

    def one_step(self, action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](
            self.shape, self.anchor, self.board)

        cleared_lines = 0
        done = False
        if self._has_dropped():
            self._set_piece(True)
            cleared_lines += self._clear_lines()

            # game over state
            if np.any(self.board[:, 1]):
                self.clear()
                self.n_deaths += 1
                done = True
            else:
                self._new_piece()

        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state, cleared_lines, done

    def clear(self):
        self.score = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)

        return self.board

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i),
                           int(self.anchor[1] + j)] = on

    def __repr__(self):
        return "ENGINE_%s_%s_%s_%s_%s" % (self.clear_line_reward,
                                             self.height_penalty,
                                             self.hole_penalty,
                                             self.bumpiness_penalty,
                                             self.game_over_penalty,
                                             )

    def calculate_score(self, board, cleared, done):
        holes_score = self.calculate_holes(board)
        bumpiness_score, height_score = self.calculate_bumpiness_and_height(
            board)
        if done:
            game_over_score = 1
        else:
            game_over_score = 0

        return holes_score * self.hole_penalty \
            + bumpiness_score * self.bumpiness_penalty \
            + cleared * self.clear_line_reward \
            + game_over_score * self.game_over_penalty

    def calculate_holes(self, board):
        holes = 0
        for col in range(self.width):
            found_top = False
            for row in range(1, self.height):
                if board[col][row] == 1:
                    found_top = True
                if found_top & (board[col][row] == 0):
                    holes += 1
        return holes

    def calculate_bumpiness_and_height(self, board):
        bump = 0
        heights = []
        for col in range(self.width):
            found_top = False
            height_col = 0
            for row in range(1, self.height):
                if board[col][row] == 1:
                    found_top = True
                if found_top:
                    height_col += 1
            heights.append(height_col)

        for i in range(len(heights) - 1):
            bump += abs(heights[i] - heights[i + 1])

        h = sum(heights)
        return bump, h
