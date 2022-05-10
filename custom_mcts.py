import time
import numpy as np
import math
import random
from helpers import *

class MCTS:

    def __init__(self, obs, config):
        self.conf = config
        self.state = np.asarray(obs.board).reshape(config.rows, config.columns)
        self.player = obs.mark
        self.final_action = None
        self.time_limit = self.conf.timeout - 0.3
        self.root_node = (0,)
        self.tree = {
            self.root_node: {
                'state': self.state, 
                'player': self.player,
                'child': [], 
                'parent': None, 
                'node_visits': 0,
                'node_reward': 0
            }
        }
        self.parent_visits = 0

    def get_ucb(self, node_no):
        # Compute UCB

        if not self.parent_visits:
            return -1
        else:
            # Using MCTS selection equation
            value_estimate = (
                self.tree[node_no]['node_reward'] / 
                (self.tree[node_no]['node_visits'] + 1)
            )
            exploration = math.sqrt(
                np.log(self.parent_visits) / 
                (self.tree[node_no]['node_visits'] + 1)
            )
            ucb_score = value_estimate + 2 * exploration
            return ucb_score

    def selection(self):

        leaf_id = (0, )
        while True:

            # Find number of children
            child_count = len(self.tree[leaf_id]['child'])
            
            # We reached a leaf node
            if child_count == 0:
                break
            
            # Max length
            elif len(leaf_id) == 7:
                break
            
            # Find the leaf node with max UCB
            else:
                max_ucb = -math.inf
                best_action = leaf_id

                # Iterate through each child and find UCB for each
                for i in range(child_count):
                    action = self.tree[leaf_id]['child'][i]

                    # Find child
                    child_id = leaf_id + (action, )

                    # Traversing to this child is best option if UCB > max UCB
                    ucb = self.get_ucb(child_id)

                    if ucb > max_ucb:
                        # print(f"UCB for id {child_id} = {ucb}")
                        max_ucb = ucb
                        best_action = action
                
                # Update leaf id with best action based on UCB
                leaf_id = leaf_id + (best_action,)

        # Return leaf node with max UCB
        return leaf_id

    def expansion(self, leaf_id):

        # Find current state
        current_state = self.tree[leaf_id]['state']
        
        # Find player
        player_mark = self.tree[leaf_id]['player']

        # Get current connect4 board (reshape flattened array)
        current_board = np.asarray(current_state).reshape(
            self.conf.rows * self.conf.columns
        )

        # FInd all possible actions (empty pieces on the board)
        self.actions_available = [
            c for c in range(self.conf.columns) if not current_board[c]
        ]

        # See if the game has been won
        won = game_won(current_state, player_mark, self.conf)
        child_node_id = leaf_id
        optimal_action = False

        # If we can possibly take an action
        if len(self.actions_available) and not won:

            child_nodes = []

            # Iterate through each possible action
            for action in self.actions_available:

                # Take action to reach child node
                child_id = leaf_id + (action,)
                child_nodes.append(action)

                # Obtain new board after performing the action
                new_board = put_new_piece(current_state, action, player_mark, self.conf)
                self.tree[child_id] = {
                    'state': new_board, 
                    'player': player_mark,
                    'child': [],
                    'parent': leaf_id,
                    'node_visits' : 0,
                    'node_reward' : 0
                }

                # If we win the game, it is the optimal actin
                if game_won(new_board, player_mark, self.conf):
                    best_action = action
                    optimal_action = True

            # Add obtained child nodes to the leaf node (expansion)
            self.tree[leaf_id]['child'] = child_nodes
            
            # Pick best action if won
            if optimal_action:
                child_node_id = best_action
            
            # Else pick random node
            else:
                child_node_id = random.choice(child_nodes)

        # Return expanded node
        return leaf_id + (child_node_id,)

    def simulation(self, child_node_id):
        
        # Get data of board state and other player
        self.parent_visits += 1
        state = self.tree[child_node_id]['state']
        previous_player = self.tree[child_node_id]['player']

        # Find if the previous player's move won the game
        terminal_state = game_won(state, previous_player, self.conf)
        winner = previous_player
        count = 0

        while not terminal_state:
            
            # Update board
            current_board = np.asarray(state).reshape(self.conf.rows * self.conf.columns)

            # Get all possible actions (empty squares on the board)
            self.actions_available = [c for c in range(self.conf.columns) if not current_board[c]]

            # If nobody can possibly win
            if not len(self.actions_available) or count == 3:
                winner = None
                terminal_state = True

            else:
                count += 1
                # Find next player to move
                if previous_player == 1:
                    curr_player = 2
                else:
                    curr_player = 1

                # Perform each possible action
                for action in self.actions_available:

                    # Perform move based on action
                    state = put_new_piece(state, action, curr_player, self.conf)

                    # If result is a win, break
                    result = game_won(state, curr_player, self.conf)
                    if result:
                        terminal_state = True
                        winner = curr_player
                        break

            # Update to next player's turn
            previous_player = curr_player

        # Return winner based on simulation
        return winner

    def backpropagation(self, child_node_id, winner):

        # Get the id corresponding to the player
        player = self.tree[(0,)]['player']

        # No reward if nobody has won the game
        if winner == None:
            reward = 0

        # If player won
        elif winner == player:
            reward = 1

        # If player lost
        else:
            reward = -20

        # Update node with backpropagation data
        node_id = child_node_id
        self.tree[node_id]['node_visits'] += 1
        self.tree[node_id]['node_reward'] += reward

    def start(self):

        # Save start time
        self.start_time = time.time()
        expansion_done = False

        # Continue for fixed amount of time
        while (time.time() - self.start_time) < self.time_limit:

            # Select the best node based on UCB
            node_id = self.selection()

            # print(f"Current selected node: {node_id}")

            # Perform expansion if we have not done it before
            if not expansion_done:
                # print(f"Expanding for {node_id}")
                node_id = self.expansion(node_id)
                expansion_done = True

            # Get winner by simulation
            winner = self.simulation(node_id)

            # Update node rewards using backpropagation
            self.backpropagation(node_id, winner)

        curr_id = (0, )
        possible_actions = self.tree[curr_id]['child']
        max_visits = -1

        # Iterate through each possible action from the current node
        for action in possible_actions:
            action = curr_id + (action, )

            # Find best possible action based on node visits in the tree
            visit = self.tree[action]['node_visits']
            if visit > max_visits:
                max_visits = visit
                best_action = action

        print(f"Best move: {best_action[1]}")
        return best_action