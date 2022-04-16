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
        self.mcts = MCTS({"num_simulations": 400})

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

        player_min_steps = board.min_steps_before_victory(player)
        opponent_min_steps = board.min_steps_before_victory(not player)
        # TODO: implement your agent and return an action for the current step.
        if (step < 12 and board.nb_walls[0] + board.nb_walls[1] == 20) or step < 5:
            try:
                (x, y) = board.get_shortest_path(player)[0]
            except NoPath:
                actions = list(board.get_actions(player))
                return random.choice(actions)
            return ("P", x, y)
        if (
            time_left >= 45 or player_min_steps > opponent_min_steps
        ) and board.nb_walls[player] > 0:
            action = self.mcts.run(board, player)
        # No more walls or time is running out
        else:
            try:
                (x, y) = board.get_shortest_path(player)[0]
            except NoPath:
                actions = list(board.get_actions(player))
                return random.choice(actions)

            action = ("P", x, y)

        if not board.is_action_valid(action, player):
            print("illegal: ", action)
            actions = list(board.get_actions(player))
            return random.choice(actions)

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

    def wall_in_path(self, x, y, shortest_path):
        if not len(shortest_path):
            return False
        return (
            (x, y) in shortest_path
            or (x + 1, y) in shortest_path
            or (x, y + 1) in shortest_path
            or (x + 1, y + 1) in shortest_path
            or (x - 1, y) in shortest_path
            or (x, y - 1) in shortest_path
            or (x - 1, y - 1) in shortest_path
            or (x - 1, y + 1) in shortest_path
            or (x + 1, y - 1) in shortest_path
        )

    def get_valid_wall_moves(self, state: Board, player):
        best_wall_moves = []
        opponent_wall_moves = []
        opponent = not player

        position_player = state.pawns[player]
        position_opponent = state.pawns[opponent]
        try:
            opponent_path = state.get_shortest_path(opponent)
        except:
            opponent_path = []
        try:
            player_path = state.get_shortest_path(player)
        except:
            player_path = []

        all_wall_moves = state.get_legal_wall_moves(player)

        for wall_move in all_wall_moves:
            (_, x, y) = wall_move

            distance_from_opponent = self.manhattan([x, y], position_opponent)
            distance_from_player = self.manhattan([x, y], position_player)

            if (
                distance_from_opponent <= 3
                or self.wall_in_path(x, y, opponent_path)
                or distance_from_player <= 3
                or self.wall_in_path(x, y, player_path)
            ):
                best_wall_moves.append(wall_move)

        return best_wall_moves, opponent_wall_moves

    def get_valid_actions(self, state: Board, player):
        # all_actions = state.get_actions(player)
        if state.is_finished():
            return []

        all_moves = []

        best_wall_moves, _ = self.get_valid_wall_moves(state, player)

        all_pawn_moves = state.get_legal_pawn_moves(player)

        all_moves.extend(all_pawn_moves)

        all_moves.extend(best_wall_moves)

        return all_moves

    def get_random_action(self, state: Board, player):
        pawn_moves = []
        wall_moves = []

        best_wall_moves, _ = self.get_valid_wall_moves(state, player)
        all_pawn_moves = state.get_legal_pawn_moves(player)

        pawn_moves.extend(all_pawn_moves)
        wall_moves.extend(best_wall_moves)

        if not len(pawn_moves) and not len(wall_moves):
            return None
        if len(pawn_moves):
            pawn_move=random.choice(pawn_moves)
            if len(wall_moves):
                wall_move = random.choice(wall_moves)
                return random.choices([pawn_move, wall_move], weights=(1, 20), k=1)[0]
            else:
                return pawn_move

    # Evaluates the next move to play by the agent
    def evaluate(self, state: Board, player):
        opponent = not player

        my_score = 50 * pow(state.get_score(player) - 1, 3)

        if state.pawns[player][0] == state.goals[player]:
            return 999999999
        elif state.pawns[opponent][0] == state.goals[opponent]:
            return -999999999
        if state.nb_walls[opponent] == 0 and my_score > 0:
            return 999999999
        if state.nb_walls[player] == 0 and my_score < 0:
            return -999999999

        try:
            opponent_path = state.get_shortest_path(opponent)
            if len(opponent_path) == 1:
                my_score -= 1000
            my_score += 16 * (state.pawns[opponent][1] - opponent_path[-1][1]) ** 2
            if not (state.pawns[player][1] - state.get_shortest_path(player)[-1][1]):
                my_score += 40
        except NoPath:
            pass

        my_score += 50 * (state.nb_walls[player] ** 2 - state.nb_walls[opponent] ** 2)

        return my_score

    def get_reward(self, state: Board, player, depth):
        if state.pawns[player][0] == state.goals[player]:
            return 999999999
        elif state.pawns[not player][0] == state.goals[not player]:
            return -999999999

        if depth > 2:
            return self.evaluate(state, player)

        return 0

    def run(self, state, to_play):

        root = Node(state, 0, to_play)

        # EXPAND root
        valid_moves = self.get_valid_actions(state, to_play)
        action_probs = {move: 0 for move in valid_moves}
        root.expand(action_probs)

        for i in range(self.args["num_simulations"]):
            if i % 10 == 0:
                print(i)
            next_node = root
            search_path = [next_node]


            # SELECT
            depth = 0
            while next_node.expanded():
                depth += 1
                _, next_node = next_node.select_child()
                search_path.append(next_node)

            reward = self.get_reward(
                state=next_node.state, player=root.to_play, depth=depth
            )
            if not reward:
                # If the game has not ended:
                # EXPAND
                if next_node.visit_count > 0:
                    valid_moves = self.get_valid_actions(
                        next_node.state, next_node.to_play
                    )
                    action_probs = {move: 0 for move in valid_moves}
                    next_node.expand(action_probs)

                # SIMULATION
                simulation_state = next_node.state.clone()
                simulation_player = next_node.to_play
                while not reward:
                    depth += 1
                    reward = self.get_reward(
                        simulation_state, player=root.to_play, depth=depth
                    )
                    random_action = self.get_random_action(
                        simulation_state, simulation_player
                    )
                    if random_action is not None:
                        simulation_state = simulation_state.play_action(
                            random_action, simulation_player,
                        )
                        simulation_player = (simulation_player + 1) % 2

                self.backpropagate(search_path, reward, next_node.to_play)
            else:
                self.backpropagate(search_path, reward, next_node.to_play)

            # print([child.value() for child in search_path])
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
    def __init__(self, state, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = state

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def next_state(self, action):
        state_clone = self.state.clone()
        next_state = state_clone.play_action(action, self.to_play)

        return next_state

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
        if not parent.value() or not child.value():
            return 0

        prior_score = 2 * math.sqrt(math.log(parent.visit_count)) / (child.visit_count)

        return child.value() + prior_score

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -math.inf
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

        # print([ child.visit_count for _, child in self.children.items()])
        return best_action

    def expand(self, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        for action, prob in action_probs.items():
            self.children[action] = Node(
                self.next_state(action), prior=prob, to_play=(self.to_play + 1) % 2
            )

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
