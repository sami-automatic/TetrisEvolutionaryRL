import curses
import numpy as np
import os
from engine_DQN import TetrisEngine


def init():
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    #curses.halfdelay(999)
    # Enumerate keys
    stdscr.keypad(True)

width, height = 6, 10
env = TetrisEngine(width, height)
done = False

while True:
    while not done:
        action_code = input("action code : ")
        state, reward, done, cleared = env.step(int(action_code))
        print(str(reward))
        print(str(state))

    print("GAME_OVER")
    done = False

'''
def play_game():
    # Store play information
    db = []
    # Initial rendering
    stdscr.addstr(str(env))

    done = False
    # Global action
    action = 6
    consecutive_actions = []
    while not done:
        action = 6
        key = stdscr.getch()
        touches_floor = False

        if key == -1:  # No key pressed
            action = 6
        elif key == ord('a'):  # Shift left
            action = 0
            consecutive_actions.append(0)
        elif key == ord('d'):  # Shift right
            action = 1
            consecutive_actions.append(1)
        elif key == ord('w'):  # Hard drop
            touches_floor = True
            action = 2
        elif key == ord('s'):  # Soft drop
            action = 3
        elif key == ord('q'):  # Rotate left
            action = 4
            consecutive_actions.append(4)
        elif key == ord('e'):  # Rotate right
            action = 5
            consecutive_actions.append(5)

        # Game step
        if touches_floor:
            consecutive_actions.append(2)
            try:
                actions_code = env.actions.index(consecutive_actions)
                state, reward, done, cleared = env.step(actions_code)
                stdscr.clear()
                stdscr.addstr(str(state))
                stdscr.addstr('reward: ' + str(reward))
            except:
                print("wrong code")
            consecutive_actions = []
        else:
            state, cleared_lines, done = env.one_step(action)

            #db.append((state, reward, done, action))

            # Render
            stdscr.clear()
            stdscr.addstr(str(state))


    return db



def play_again():
    # stdscr.addstr('Play Again? [y/n]')
    print('Play Again? [y/n]')
    print('> ', end='')
    # stdscr.addstr('> ')
    choice = input()
    # choice = stdscr.getch()

    return True if choice.lower() == 'y' else False


def save_game():
    print('Accumulated reward: {0} | {1} moves'.format(sum([i[1] for i in db]), len(db)))
    print('Would you like to store the game info as training data? [y/n]')
    # stdscr.addstr('Would you like to store the game info as training data? [y/n]\n')
    # stdscr.addstr('> ')
    print('> ', end='')
    choice = input()
    return True if choice.lower() == 'y' else False


def terminate():
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()


def init():
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    curses.halfdelay(7)
    # Enumerate keys
    stdscr.keypad(True)

    # return stdscr


if __name__ == '__main__':
    # Curses standard screen
    stdscr = curses.initscr()

    # Init environment
    width, height = 5, 10  # standard tetris friends rules
    env = TetrisEngine(width, height)

    # Play games on repeat
    while True:
        init()
        stdscr.clear()
        env.clear()
        db = play_game()

        # Return to terminal
        terminate()
        # Should the game info be saved?
        if save_game():
            try:
                fr = open('training_data.npy', 'rb')
                x = np.load(fr)
                fr.close()
                fw = open('training_data.npy', 'wb')
                x = np.concatenate((x, db))
                # print('Saving {0} moves...'.format(len(db)))
                np.save(fw, x)
                print('{0} data points in the training set'.format(len(x)))
            except Exception as e:
                print('no training file exists. Creating one now...')
                fw = open('training_data.npy', 'wb')
                print('Saving {0} moves...'.format(len(db)))
                np.save(fw, db)
        # Prompt to play again
        if not play_again():
            print('Thanks for contributing!')
            break
'''