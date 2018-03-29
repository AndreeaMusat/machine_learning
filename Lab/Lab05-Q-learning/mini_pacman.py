# Tudor Berariu, 2016

from copy import copy
from random import choice

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
ACTION_EFFECTS = {
    "UP": (-1,0),
    "RIGHT": (0,1),
    "DOWN": (1,0),
    "LEFT": (0,-1),
    "STAY": (0,0)
}

MOVE_REWARD = -0.1
WIN_REWARD = 10.0
LOSE_REWARD = -10.0

## Functions to serialize / deserialize game states
def __serialize_state(state):
    return "\n".join(map(lambda row: "".join(row), state))

def __deserialize_state(str_state):
    return list(map(list, str_state.split("\n")))

## Return the initial state of the game
def get_initial_state(map_file_path):
    with open(map_file_path) as map_file:
        initial_str_state = map_file.read().strip()
    return initial_str_state

## Return the available actions in a given state
def get_legal_actions(str_state):
	all_actions = copy(ACTIONS)
	return all_actions

## Get the coordinates of an actor
def __get_position(state, marker):
    for row_idx, row in enumerate(state):
        if marker in row:
            return row_idx, row.index(marker)
    return -1, -1

## Check if is a final state
def is_final_state(str_state, score):
    return score < -20.0 or "G" not in str_state or "o" not in str_state

## Check if the given coordinates are valid (on map and not a wall)
def __is_valid_cell(state, row, col):
    return row >= 0 and row < len(state) and \
        col >= 0 and col < len(state[row]) and \
        state[row][col] != "*"

## Move to next state
def apply_action(str_state, action):
    assert(action in ACTIONS)
    message = "Greuceanu moved %s." % action

    state = __deserialize_state(str_state)
    g_row, g_col = __get_position(state, "G")
    assert(g_row >= 0 and g_col >= 0)             # Greuceanu must be on the map

    next_g_row = g_row + ACTION_EFFECTS[action][0]
    next_g_col = g_col + ACTION_EFFECTS[action][1]

    if not __is_valid_cell(state, next_g_row, next_g_col):
        next_g_row = g_row
        next_g_col = g_col
        message = message + " Not a valid cell there."

    state[g_row][g_col] = " "
    if state[next_g_row][next_g_col] == "B":
        message = message + " Greuceanu stepped on the balaur."
        return __serialize_state(state), LOSE_REWARD, message
    elif state[next_g_row][next_g_col] == "o":
        state[next_g_row][next_g_col] = "G"
        message = message + " Greuceanu found 'marul fermecat'."
        return __serialize_state(state), WIN_REWARD, message
    state[next_g_row][next_g_col] = "G"

    ## Balaur moves now
    b_row, b_col = __get_position(state, "B")
    assert(b_row >= 0 and b_col >= 0)

    dy, dx = next_g_row - b_row, next_g_col - b_col

    is_good = lambda dr, dc:__is_valid_cell(state, b_row + dr, b_col + dc)

    next_b_row, next_b_col = b_row, b_col
    if abs(dy) > abs(dx) and is_good(int(dy / abs(dy)), 0):
        next_b_row = b_row + int(dy / abs(dy))
    elif abs(dx) > abs(dy) and is_good(0, int(dx / abs(dx))):
        next_b_col = b_col + int(dx / abs(dx))
    else:
        options = []
        if abs(dx) > 0:
            if is_good(0, int(dx / abs(dx))):
                options.append((b_row, b_col + int(dx / abs(dx))))
        else:
            if is_good(0, -1):
                options.append((b_row, b_col - 1))
            if is_good(0, 1):
                options.append((b_row, b_col + 1))
        if abs(dy) > 0:
            if is_good(int(dy / abs(dy)), 0):
                options.append((b_row + int(dy / abs(dy)), b_col))
        else:
            if is_good(-1, 0):
                options.append((b_row - 1, b_col))
            if is_good(1, 0):
                options.append((b_row + 1, b_col))

        if len(options) > 0:
            next_b_row, next_b_col = choice(options)

    if state[next_b_row][next_b_col] == "G":
        message = message + " The balaur ate Greuceanu."
        reward = LOSE_REWARD
    elif state[next_b_row][next_b_col] == "o":
        message = message + " The balaur found marul fermecat. Greuceanu lost!"
        reward = LOSE_REWARD
    else:
        message = message + " The balaur follows Greuceanu."
        reward = MOVE_REWARD

    state[b_row][b_col] = " "
    state[next_b_row][next_b_col] = "B"

    return __serialize_state(state), reward, message

def display_state(state):
    print(state)
