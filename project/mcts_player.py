#!/usr/bin/env python3
"""
Quoridor agent.
Copyright (C) 2013, <<<<<<<<<<< YOUR NAMES HERE >>>>>>>>>>>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

"""

from quoridor import *

import math
import time
import numpy as np


class MyAgent(Agent):

    """My Quoridor agent."""
    def __init__(self):
        self.mcts = MCTS({"num_simulations" : 10000})

    def play(self, percepts, player, step, time_left):
        """
        This function is used to play a move according
        to the percepts, player and time left provided as input.
        It must return an action representing the move the player
        will perform.
        :param percepts: dictionary representing the current board
            in a form that can be fed to `dict_to_board()` in quoridor.py.
        :param player: the player to control in this step (0 or 1)
        :param step: the current step number, starting from 1
        :param time_left: a float giving the number of seconds left from the time
            credit. If the game is not time-limited, time_left is None.
        :return: an action
          eg: ('P', 5, 2) to move your pawn to cell (5,2)
          eg: ('WH', 5, 2) to put a horizontal wall on corridor (5,2)
          for more details, see `Board.get_actions()` in quoridor.py
        """
        print("percept:", percepts)
        print("player:", player)
        print("step:", step)
        print("time left:", time_left if time_left else "+inf")

        board = dict_to_board(percepts)

        # TODO: implement your agent and return an action for the current step.
        if time_left >= 45 and board.nb_walls[player] > 0:
            action = self.mcts.run(board, player)
            print(action)
        # No more walls or time is running out
        else:
            try:
                (x, y) = board.get_shortest_path(player)[0]
            except NoPath:
                print("NO PATH")
                return None

            action = ("P", x, y)

        return action

    def cutoff(self, step, depth, start_time, time_left):
        current_time = time.time()
        # 5 seconds to search
        if current_time - start_time >= 5:
            return True
        # Reduce depth at the start or end of the game
        if step < 7 or time_left < 100:
            return depth >= 2
        return depth > 25


class MCTS:
    def __init__(self, args):
        self.args = args

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_valid_wall_moves(self, wall_moves, state: Board, other_player):
        best_wall_moves = []
        position_opponent = state.pawns[other_player]
        for wall_move in wall_moves:
            (_, x, y) = wall_move
            # Walls close to opponent
            position_from_opponent = self.manhattan([x, y], position_opponent)
            if position_from_opponent <= 3:
                best_wall_moves.append(wall_move)
        return best_wall_moves

    def get_valid_actions(self, state: Board, player):
        # all_actions = state.get_actions(player)
        actions_to_explore = []
        all_pawn_moves = state.get_legal_pawn_moves(player)
        all_wall_moves = state.get_legal_wall_moves(player)

        opponent = (player + 1) % 2

        actions_to_explore.extend(
            self.get_valid_wall_moves(all_wall_moves, state, opponent)
        )
        actions_to_explore.extend(all_pawn_moves)

        return actions_to_explore

    def get_reward(self, state: Board, player):
        opponent = (player + 1) % 2

        reward = int(state.pawns[player][0] == state.goals[player])

        reward += int(state.pawns[opponent][0] == state.goals[opponent]) * -1

        return reward

    def run(self, state, to_play):

        root = Node(0, to_play)

        # EXPAND root
        valid_moves = self.get_valid_actions(state, to_play)
        action_probs = { move : 0 for move in valid_moves }
        root.expand(state.clone(), to_play, action_probs)

        for _ in range(self.args["num_simulations"]):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = parent.state
            state_clone = state.clone()
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state = state_clone.play_action(action, parent.to_play)

            # The value of the new state from the perspective of the other player
            value = self.get_reward(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                valid_moves = self.get_valid_actions(state, to_play)
                action_probs = { move : 0 for move in valid_moves }
                node.expand(next_state, (parent.to_play + 1) % 2, action_probs)

            self.backpropagate(search_path, value, (parent.to_play + 1) % 2)

        return root.next_move()

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1


class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def ucb_score(self, parent, child):
        """
        The score for an action that would transition between the parent and child.
        """
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        if child.visit_count > 0:
            # The value of the child is from the perspective of the opposing player
            value_score = -child.value()
        else:
            value_score = 0

        return value_score + prior_score

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = self.ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def next_move(self):
        """
        Select the child with the highest UCB score.
        """
        max_visits = -np.inf
        best_action = None

        for action, child in self.children.items():
            visit_count = child.visit_count
            if visit_count > max_visits:
                max_visits = visit_count
                best_action = action

        print(self.children)
        return best_action

    def expand(self, state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.state = state
        for action, prob in action_probs.items():
            self.children[action] = Node(prior=prob, to_play=(self.to_play + 1) % 2)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(
            self.state.__str__(), prior, self.visit_count, self.value()
        )


if __name__ == "__main__":
    agent_main(MyAgent())
