# GETS STATE, RETURNS ACTION LIST
from engineLittle import TetrisEngine
from math import floor

# Toggle this to true in order to follow output
debug = False

COLUMN_COUNT = 20
ROW_COUNT = 5
mid = floor(ROW_COUNT / 2)  # middle of the rows.

TOUCHES_ANOTHER_BLOCK_REWARD = 6.0
TOUCHES_FLOOR_REWARD = 10.0
TOUCHES_WALL_REWARD = 5.0
CLEAR_LINE_REWARD = 150.0

HEIGHT_MULTIPLIER_PENALTY = -0.03
HOLE_PENALTY = -0.01
BLOCKADE_PENALTY = -0.3
BUMPINESS_PENALTY = -0.2

env = TetrisEngine(ROW_COUNT, COLUMN_COUNT)
obs = env.clear()

def calculate_edge(obs, i, j):
    point = 0.0     # cumulative point of edge
    neighbours = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]   # neighbours of cell.
    got_floor_point = False
    for (x, y) in neighbours:
        if j == COLUMN_COUNT - 1 and not got_floor_point:   # touching the floor
            debug and print("Touching the floor, 5 point!")
            point += TOUCHES_FLOOR_REWARD
            got_floor_point = True
        try:                # has neighboring index
            if obs[x][y]:   # touches another block
                debug and print("Touching another block, 3 point! Touches: (%d, %d)" % (x, y))
                point += TOUCHES_ANOTHER_BLOCK_REWARD
        except IndexError:                             # point is on the edge
            if y == COLUMN_COUNT and got_floor_point:  # if we added the floor point, don't add extra 2.5!
                debug and print("Floor point already added.")
                continue
            debug and print("Is on edge, 2.5 point! Cant find: (%d, %d)" % (x, y))
            point += TOUCHES_WALL_REWARD
    return point

def calculate_hole_score(hole_count):
    return HOLE_PENALTY * hole_count

def get_highest_index(list):
    index = 0    
    for (i, value) in enumerate(list):
        if value == 1 and i > index:
            index = i
    return index

def calculate_bumpiness_score(obs):
    penalty = 0.0
    for i in range(ROW_COUNT - 1):
        diff = abs(get_highest_index(obs[i]) - get_highest_index(obs[i + 1]))
        penalty += diff * BUMPINESS_PENALTY
    return penalty

def is_starting_cell(i, j):
    # (i == mid or i == (mid - 1))
    return j == 0 and i == 0

def calculate_blockaded_score(blockaded_count):
    return BLOCKADE_PENALTY * blockaded_count

def calculate_cleared_score(cleared):
    return CLEAR_LINE_REWARD * cleared

def calculate_score(obs, cleared):
    score = 0.0
    hole_count = 0
    blockaded_count = 0
    for (i, row) in enumerate(obs):
        for (j, cell) in enumerate(row):
            if is_starting_cell(i, j):   # starting position of block. ignore!
                debug and print("Starting position, skipping..")
                continue
            if cell:
                debug and print("i:", i, "\tj:", j, "\tvalue:", cell)
                edge_point = calculate_edge(obs, i, j)
                debug and print("Point:", edge_point)
                score += edge_point
            else:
                try:
                    if obs[i - 1][j]:   # empty and top of blocked
                        blockaded_count += 1
                    else:               # top is not blocked, normal hole
                        hole_count += 1
                except IndexError:      # top doesn't exist, normal hole
                    hole_count += 1
    debug and print("Score:", score)
    debug and print("Empty row count:", hole_count)
    total_score, hole_score, bumpiness_score, blockaded_score = (0, 0, 0, 0)
    cleared_score = calculate_cleared_score(cleared)
    if cleared == 0:
        hole_score = calculate_hole_score(hole_count)
        bumpiness_score = calculate_bumpiness_score(obs)
        blockaded_score = calculate_blockaded_score(blockaded_count)
    debug and print("Empty cell score:", hole_score)
    debug and print("Bumpiness score:", bumpiness_score)
    debug and print("Blockaded score:", blockaded_score)
    print("Cleared score:", cleared_score)
    total_score = score + hole_score + bumpiness_score + blockaded_score + cleared_score
    total_score = round(total_score, 2)
    print("Total score:", total_score, "\n\n")

# 0: left,
# 1: right,
# 2: hard_drop,
# 3: soft_drop,
# 4: rotate_left,
# 5: rotate_right,
# 6: idle

# done = False
# while done is False:
#     obs, reward, done, cleared = env.step(2)
#     debug and print("obs: \n", obs, "\nreward:",reward, "\ndone:", done, "\n\n")
#     calculate_score(obs, cleared)

for i in range(5):
    for j in range(i):
        debug and print("go right")
        obs, reward, done, cleared = env.step(1)    # right
        debug and print("obs: \n", obs, "\nreward:",reward, "\ndone:", done, "\n\n")
    debug and print("go down")
    obs, reward, done, cleared = env.step(2)        # down
    print("obs: \n", obs, "\ndone:", done)
    calculate_score(obs, cleared)

print("\n============\nFinal state\n============\nobs: \n", obs, "\nreward:",reward, "\ndone:", done, "\n\n")