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


class MyAgent(Agent):

    """My Quoridor agent."""

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

        player_min_steps = board.min_steps_before_victory(player)
        opponent_min_steps = board.min_steps_before_victory(not player)
        if (
            time_left >= 45 or player_min_steps > opponent_min_steps
        ) and board.nb_walls[player] > 0:
            print(time_left)
            value, action = self.minimax(board, player, step, time_left)
            print(value, action)
        # No more walls or time is running out
        else:
            try:
                (x, y) = board.get_shortest_path(player)[0]
            except NoPath:
                print("NO PATH")
                return None

            action = ("P", x, y)

        return action

    def treshold(self, step, depth, start_time):
        current_time = time.time()

        if current_time - start_time >= 5:
            return True

        if step <= 7:
            return depth >= 2

        return depth > 25

    def minimax(self, state: Board, player, step, time_left):
        start = time.time()
        return self.max_value(
            state, player, step, start, time_left, -float("inf"), float("inf"), 0
        )

    def max_value(
        self, state: Board, player, step, start_time, time_left, alpha, beta, depth: int
    ):
        if self.treshold(step, depth, start_time):
            return self.evaluate(state, player), None

        if state.is_finished():
            return state.get_score(player), None

        v_star = -math.inf
        m_star = None
        for action in self.get_actions(state, player):
            clone = state.clone()
            clone.play_action(action, player)
            next_state = clone
            v, _ = self.min_value(
                next_state, player, step, start_time, time_left, alpha, beta, depth + 1
            )
            if v > v_star:
                v_star = v
                m_star = action
                alpha = max(alpha, v_star)
            if v >= beta:
                return v_star, m_star
        return v_star, m_star

    def min_value(
        self, state: Board, player, step, start_time, time_left, alpha, beta, depth: int
    ):
        if self.treshold(step, depth, start_time):
            return self.evaluate(state, player), None

        if state.is_finished():
            return state.get_score(player), None

        v_star = math.inf
        m_star = None
        for action in self.get_actions(state, player):
            clone = state.clone()
            clone.play_action(action, player)
            next_state = clone
            v, _ = self.max_value(
                next_state, player, step, start_time, time_left, alpha, beta, depth + 1
            )
            if v < v_star:
                v_star = v
                m_star = action
                beta = min(beta, v_star)
            if v <= alpha:
                return v_star, m_star
        return v_star, m_star

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_wall_moves(self, state: Board, player):

        best_wall_moves = []
        opponent_wall_moves = []
        position_player = state.pawns[player]
        position_opponent = state.pawns[not player]
        all_wall_moves = state.get_legal_wall_moves(player)

        for wall_move in all_wall_moves:
            (_, x, y) = wall_move
            # Walls close to opponent
            distance_from_opponent = self.manhattan([x, y], position_opponent)
            distance_from_player = self.manhattan([x, y], position_player)

            if distance_from_opponent <= 3:
                opponent_wall_moves.append(wall_move)
                best_wall_moves.append(wall_move)
            if distance_from_player <= 3:
                best_wall_moves.append(wall_move)

        return best_wall_moves, opponent_wall_moves

    def get_actions(self, state: Board, player):
        # all_actions = state.get_actions(player)
        actions_to_explore = []
        all_pawn_moves = state.get_legal_pawn_moves(player)

        best_wall_moves, _ = self.get_wall_moves(state, player)

        actions_to_explore.extend(best_wall_moves)

        actions_to_explore.extend(all_pawn_moves)

        return actions_to_explore

    def evaluate(self, state: Board, player):
        opponent = not player

        try:
            player_min_steps = state.min_steps_before_victory(player)
            opponent_min_steps = state.min_steps_before_victory(opponent)
            my_score = 100 / max(player_min_steps, 0.001)
            my_score -= 100 / (max(opponent_min_steps, 0.01) ** 2)
        except NoPath:
            print("NO PATH estimate_score")
            return -float("inf")

        my_score += (state.nb_walls[player]) - state.nb_walls[opponent]

        my_score += state.pawns[opponent][1] - state.get_shortest_path(opponent)[-1][1]
        # my_score -= state.pawns[player][1] - state.get_shortest_path(player)[-1][1]

        if state.pawns[player][0] == state.goals[player]:
            return float("inf")
        elif state.pawns[opponent][0] == state.goals[opponent]:
            return -float("inf")
        # print(my_score)

        return my_score


if __name__ == "__main__":
    agent_main(MyAgent())
