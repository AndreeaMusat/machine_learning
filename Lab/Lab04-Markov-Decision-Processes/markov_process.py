import sys
import os.path
import numpy as np
from argparse import ArgumentParser
from copy import copy
from random import choice

INF = 99999999999999999999999

class Maze:

    NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3  # actions

    DYNAMICS = {  # the stochastic effects of actions
        NORTH: {(0, -1): 0.1, (-1, 0): .8, (0, 1): .1},
        EAST: {(-1, 0): 0.1, (0, 1): .8, (1, 0): .1},
        SOUTH: {(0, 1): 0.1, (1, 0): .8, (0, -1): .1},
        WEST: {(1, 0): 0.1, (0, -1): .8, (-1, 0): .1},
    }

    WALL, EMPTY = "x", " "

    VISUALS = {
        (0, 0, 1, 1): "\N{BOX DRAWINGS HEAVY DOWN AND RIGHT}",
        (1, 0, 0, 1): "\N{BOX DRAWINGS HEAVY DOWN AND LEFT}",
        (1, 0, 1, 0): "\N{BOX DRAWINGS HEAVY HORIZONTAL}",
        (0, 1, 1, 0): "\N{BOX DRAWINGS HEAVY UP AND RIGHT}",
        (1, 1, 0, 0): "\N{BOX DRAWINGS HEAVY UP AND LEFT}",
        (0, 1, 0, 1): "\N{BOX DRAWINGS HEAVY VERTICAL}",
        (1, 1, 1, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND HORIZONTAL}",
        (1, 1, 1, 0): "\N{BOX DRAWINGS HEAVY UP AND HORIZONTAL}",
        (1, 1, 0, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND LEFT}",
        (1, 0, 1, 1): "\N{BOX DRAWINGS HEAVY DOWN AND HORIZONTAL}",
        (0, 1, 1, 1): "\N{BOX DRAWINGS HEAVY VERTICAL AND RIGHT}",
        (1, 0, 0, 0): "\N{BOX DRAWINGS HEAVY LEFT}",
        (0, 1, 0, 0): "\N{BOX DRAWINGS HEAVY UP}",
        (0, 0, 1, 0): "\N{BOX DRAWINGS HEAVY RIGHT}",
        (0, 0, 0, 1): "\N{BOX DRAWINGS HEAVY DOWN}",
        (0, 0, 0, 0): "\N{BOX DRAWINGS HEAVY VERTICAL AND HORIZONTAL}",
        WEST: "\N{LEFTWARDS ARROW}",
        NORTH: "\N{UPWARDS ARROW}",
        EAST: "\N{RIGHTWARDS ARROW}",
        SOUTH: "\N{DOWNWARDS ARROW}",
    }

    def __init__(self, map_name):
        self._rewards, self._cells = {}, []
        with open(os.path.join("maps", map_name), "r") as map_file:
            for line in map_file.readlines():
                if ":" in line:
                    name, value = line.strip().split(":")
                    self._rewards[name] = float(value)
                else:
                    self._cells.append(list(line.strip()))
        self._states = [(i, j) for i, row in enumerate(self._cells)
                        for j, cell in enumerate(row) if cell != Maze.WALL]

    @property
    def actions(self):
        return [Maze.NORTH, Maze.EAST, Maze.SOUTH, Maze.WEST]

    @property
    def states(self):
        return copy(self._states)

    def is_final(self, state):
        row, col = state
        return self._cells[row][col] != Maze.EMPTY

    def effects(self, state, action):
        """
        Returns a list of tuples (s', p, r) (new_state, probability, reward)
        """
        if self.is_final(state):
            return []
        row, col = state
        next_states = {}
        for (d_row, d_col), prob in Maze.DYNAMICS[action].items():
            next_row, next_col = row + d_row, col + d_col
            if self._cells[next_row][next_col] == Maze.WALL:
                next_row, next_col = row, col
            if (next_row, next_col) in next_states:
                prev_prob, _ = next_states[(next_row, next_col)]
                prob += prev_prob
            cell = self._cells[next_row][next_col]
            reward = self._rewards["default" if cell == Maze.EMPTY else cell]
            next_states[(next_row, next_col)] = (prob, reward)
        return [(s, p, r) for s, (p, r) in next_states.items()]

    def print_policy(self, policy):
        last_row = []
        height = len(self._cells)

        for row, row_cells in enumerate(self._cells):
            width = len(row_cells)
            for col, cell in enumerate(row_cells):
                if cell == Maze.WALL:
                    north, south, west, east = 0, 0, 0, 0
                    if last_row and len(last_row) > col:
                        north = last_row[col] == Maze.WALL
                    if row + 1 < height:
                        south = self._cells[row + 1][col] == Maze.WALL
                    if col > 0:
                        west = row_cells[col - 1] == Maze.WALL
                    if col + 1 < width:
                        east = row_cells[col + 1] == Maze.WALL
                    sys.stdout.write(Maze.VISUALS[(west, north, east, south)])
                elif self.is_final((row, col)):
                    sys.stdout.write(cell)
                else:
                    action = policy[(row, col)]
                    sys.stdout.write(Maze.VISUALS[action])
            sys.stdout.write("\n")
            last_row = row_cells
        sys.stdout.flush()

    def print_values(self, v):
        for r, row_cells in enumerate(self._cells):
            print(" | ".join(["{0:5.2f}".format(v[r, c]) if cell == Maze.EMPTY else "====="
                              for c, cell in enumerate(row_cells)]))



def policy_iteration(game, args):
    gamma = args.gamma
    max_delta = args.max_delta

    # the value function is initialized with zeros
    v = {s: 0 for s in game.states}

    # the policy is initialized using random values
    policy = {s: choice(game.actions)
              for s in game.states if not game.is_final(s)}

    ######################## TODO 1 ############################
    non_terminal_states = [s for s in game.states if not game.is_final(s)]

    while True:

        # evaluate the current policy
        while True:
            curr_delta = 0  # used for convergence tests; = max_s (abs(v_new[s] - v_old[s])) 
            for s in non_terminal_states:
                v_old, v_new = v[s], 0

                # v_new = sum_{s_prime} [p(s', r | s, a) * (r + gamma * v[s_prime])]
                for i in range(len(game.effects(s, policy[s]))):
                    s_prime, prob, reward = game.effects(s, policy[s])[i]
                    v_new += prob * (reward + gamma * v[s_prime])

                curr_delta = max(curr_delta, abs(v_new - v_old))
                v[s] = v_new

            if curr_delta < max_delta:
                break

        # find a better policy
        done = True
        for s in non_terminal_states:
            old_action, new_action = policy[s], policy[s]
            best_action_eval = -INF
            
            for action in game.actions:
                # apply the current policy and greedily find the action with the highest long term discounted reward
                curr_action_eval = 0

                for i in range(len(game.effects(s, action))):
                    s_prime, prob, reward = game.effects(s, action)[i]
                    curr_action_eval += prob * (reward + gamma * v[s_prime])

                # update the best action if an action with a higher reward was found
                if curr_action_eval > best_action_eval:
                    best_action_eval = curr_action_eval
                    new_action = action

            # update the policy
            policy[s] = new_action

            # convergence test
            done = done and (new_action == old_action)

        if done:
            break
    ############################################################
    return policy, v

def value_iteration(game, args):
    gamma = args.gamma
    max_delta = args.max_delta
    
    # only initialize the value function
    v = {s: 0 for s in game.states}

    ######################## TODO 2 ############################
    non_terminal_states = [s for s in game.states if not game.is_final(s)]
    while True:
        curr_delta = 0
        for s in non_terminal_states:
            v_old = v[s]
            max_v_new = -INF
            max_v_new_action = None

            # instead of computing the value function for the current policy, 
            # compute the value function for every action and keep the one with the 
            # highest value
            for action in game.actions:
                curr_v = 0
                for i in range(len(game.effects(s, action))):
                    s_prime, prob, reward = game.effects(s, action)[i]
                    curr_v += prob * (reward + gamma * v[s_prime])
                if curr_v > max_v_new:
                    max_v_new = curr_v
                    max_v_new_action = action

            v[s] = max_v_new
            curr_delta = max(curr_delta, abs(v[s] - v_old))

        # convergence test
        if curr_delta < max_delta:
            break

    policy = {}
    for s in non_terminal_states:
        best_action = None
        best_action_eval = -INF

        for action in game.actions:
            curr_action_eval = 0
            for i in range(len(game.effects(s, action))):
                s_prime, prob, reward = game.effects(s, action)[i]
                curr_action_eval += prob * (reward + gamma * v[s_prime])

            if curr_action_eval > best_action_eval:
                best_action_eval = curr_action_eval
                best_action = action

        if best_action is not None:
            policy[s] = best_action

    ############################################################
    return policy, v

################# test policy ##################################
def test_policy(game, policy, state):
    print(state)
    while not game.is_final(state):
        effects = game.effects(state, policy[state])
        new_states = [eff[0] for eff in effects]
        new_states_probs = [eff[1] for eff in effects]
        num_states = len(effects)
        new_state_idx = np.random.choice(np.arange(num_states), p=new_states_probs)
        state = new_states[new_state_idx]
        print(state)
################################################################

def main():
    argparser = ArgumentParser()
    argparser.add_argument("--map-name", type=str, default="simple")
    argparser.add_argument("--gamma", type=float, default=.9)
    argparser.add_argument("--max-delta", type=float, default=1e-8)

    args = argparser.parse_args()
    game = Maze(args.map_name)

    print("Policy iteration:")
    policy, v = policy_iteration(game, args)
    game.print_values(v)
    game.print_policy(policy)

    # make a simulation
    # test_policy(game, policy, (4, 4))
   
    print("Value iteration:")
    policy, v = value_iteration(game, args)
    game.print_values(v)
    game.print_policy(policy)

    # another simulation
    # test_policy(game, policy, (4, 4))

if __name__ == "__main__":
    main()
