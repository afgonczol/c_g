# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:54:15 2019

@author: c62wt96
"""

import numpy as np
import sys
import copy
import time
from functools import lru_cache
import pickle
import math
import random


turn = 0
board = np.zeros((9, 9)).astype("int8")
meta_board = np.zeros((3, 3)).astype("int8")

winning_boards = [np.zeros((3, 3)).astype("int8") for i in range(8)]
winning_boards[0][:, 0] = 1
winning_boards[1][:, 1] = 1
winning_boards[2][:, 2] = 1

winning_boards[3][0, :] = 1
winning_boards[4][1, :] = 1
winning_boards[5][2, :] = 1

winning_boards[6][0, 0] = 1
winning_boards[6][1, 1] = 1
winning_boards[6][2, 2] = 1

winning_boards[7][0, 2] = 1
winning_boards[7][1, 1] = 1
winning_boards[7][2, 0] = 1

winning_boards_flat = [x.flatten() for x in winning_boards]


@lru_cache(maxsize=2 ** 14)
def hash_winning(mini_board):
    mini_board = np.array(mini_board, dtype="int8")
    for wb in winning_boards_flat:
        result = np.sum(mini_board * wb)
        if result == 3:
            return True, 1
        elif result == -3:
            return True, -1

    return False, False  # Second variable isn't used in this case


@lru_cache(maxsize=2 ** 14)
def mini_score(mini_board):  # input is flat tuple
    mini_board = np.array(mini_board, dtype="int8")
    score = np.sum(mini_board)
    score += mini_board[4] * 0.7
    score += (mini_board[0] + mini_board[2] + mini_board[6] + mini_board[8]) * 0.3

    potentials = np.where(mini_board == 0)

    for move in potentials:
        for piece in [1, -1]:
            copy_board = copy.deepcopy(mini_board)
            copy_board[move] = piece
            result, r_piece = hash_winning(tuple(copy_board))
            if result:
                score += r_piece * 11

    return score


class Game:
    def __init__(self, board=None, meta_board=None, last_move=None):
        self.board = np.zeros((9, 9)).astype("int8")
        self.meta_board = np.zeros((3, 3)).astype("int8")
        self.last_move = last_move

    def reset(self):
        self.board = np.zeros((9, 9)).astype("int8")
        self.meta_board = np.zeros((3, 3)).astype("int8")
        self.last_move = None

    # add functions for mcts
    def getPossibleActions(self):
        return self.available_moves()

    def takeAction(self, action):
        self.make_move(action)
        return self

    def isTerminal(self):
        return self.is_terminal()

    def getReward(self):
        return (
            self.score() * self.last_move[2]
        )  # For MCTS, score needs to be interpreted as "higher = better" for both players

    def get_winner(self):
        if self.is_terminal():
            return min(1, max(-1, self.score()))
        else:
            return 0

    def make_move(self, move):
        # Assumes move is valid
        row, col, piece = move
        meta_row = row // 3
        meta_col = col // 3
        self.board[row, col] = piece

        if self.meta_board[meta_row, meta_col] == 0:
            mini_board = self.board[
                meta_row * 3 : meta_row * 3 + 3, meta_col * 3 : meta_col * 3 + 3
            ]

            result, r_piece = hash_winning(tuple(mini_board.flatten()))
            if result:
                self.meta_board[meta_row][meta_col] = r_piece
                mini_board.fill(r_piece)

        self.last_move = move

    def is_terminal(self):
        for wb in winning_boards:
            result = np.sum(self.meta_board * wb)
            if abs(result) == 3:
                return True
        return np.sum(np.where(self.board == 0)) == 0

    def _score(self):
        row, col, piece = self.last_move

        # First check overall game winner
        for wb in winning_boards:
            result = np.sum(self.meta_board * wb)
            if abs(result) == 3:
                return 9e9 * result

        # If there is no winner...
        # TODO: Improve evaluation to account for almost completing a board
        score = np.sum(self.meta_board) * 100
        score += self.meta_board[1, 1] * 20  # Count middle tile extra
        score += (
            self.meta_board[0, 0]
            + self.meta_board[0, 2]
            + self.meta_board[2, 0]
            + self.meta_board[2, 2]
        ) * 7  # Slight bonus for corners

        open_rows, open_cols = np.where(self.meta_board == 0)

        for r, c in zip(open_rows, open_cols):
            mini_board = self.board[r * 3 : r * 3 + 3, c * 3 : c * 3 + 3]
            score += (
                mini_board[1, 1] * 2
                + mini_board[0, 0]
                + mini_board[0, 2]
                + mini_board[2, 0]
                + mini_board[2, 2]
            )
        return score

    def score(self):
        row, col, piece = self.last_move
        # First check overall game winner
        result, r_piece = hash_winning(tuple(self.meta_board.flatten()))
        if result:
            return 9e9 * r_piece

        if np.sum(np.where(self.board == 0)) == 0:
            if np.sum(self.meta_board) > 0:
                return 9e9
            else:
                return -9e9

        # If there is no winner...
        # TODO: Improve evaluation to account for almost completing a board
        # score = np.sum(self.meta_board) * 100
        score = mini_score(tuple(self.meta_board.flatten())) * 15

        open_rows, open_cols = np.where(self.meta_board == 0)

        for r, c in zip(open_rows, open_cols):
            mini_board = self.board[r * 3 : r * 3 + 3, c * 3 : c * 3 + 3]
            score += mini_score(tuple(mini_board.flatten()))
        return score

    def meta_available_moves(self):
        # 0-8 for the 9 available spaces on the metaboard. Return 9 if can play anywhere
        try:
            row, col, piece = self.last_move
            meta_row = row % 3
            meta_col = col % 3
        except:
            return 9

        mini_board = self.board[
            meta_row * 3 : meta_row * 3 + 3, meta_col * 3 : meta_col * 3 + 3
        ]

        if (
            self.meta_board[meta_row, meta_col] != 0
            or np.sum(np.where(mini_board == 0)) == 0
        ):
            # Can play anywhere
            return 9

        # If metaboard not scored, must play on that miniboard
        else:
            return 3 * meta_row + meta_col

    def available_moves(self):
        try:
            row, col, piece = self.last_move
            meta_row = row % 3
            meta_col = col % 3
        except:
            rows, cols = np.where(self.board == 0)
            return [(r, c, 1) for r, c in zip(rows, cols)]

        mini_board = self.board[
            meta_row * 3 : meta_row * 3 + 3, meta_col * 3 : meta_col * 3 + 3
        ]

        if (
            self.meta_board[meta_row, meta_col] != 0
            or np.sum(np.where(mini_board == 0)) == 0
        ):
            # Can play anywhere
            rows, cols = np.where(self.board == 0)
            available_moves = [(r, c, -piece) for r, c in zip(rows, cols)]

        # If metaboard not scored, must play on that miniboard
        else:
            rows, cols = np.where(mini_board == 0)
            available_moves = [
                (3 * meta_row + r, 3 * meta_col + c, -piece) for r, c in zip(rows, cols)
            ]

        return available_moves


game = Game(board, meta_board)


def alpha_beta(node, depth, alpha, beta, max_player, best_move):
    if depth == 0 or node.is_terminal():
        return node.score(), best_move
    if max_player:
        value = -9e9
        moves = node.available_moves()
        best_move = moves[0]
        for move in moves:
            child = copy.deepcopy(node)
            child.make_move(move)
            new_value, history = alpha_beta(child, depth - 1, alpha, beta, False, move)
            if new_value > value:
                value = new_value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = 9e9
        moves = node.available_moves()
        best_move = moves[0]
        for move in moves:
            child = copy.deepcopy(node)
            child.make_move(move)
            new_value, history = alpha_beta(child, depth - 1, alpha, beta, True, move)
            if new_value < value:
                value = new_value
                best_move = move
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move
    
def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode:
    def __init__(self, state, parent):
        self.state = copy.deepcopy(state)
        self.isTerminal = self.state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts:
    def __init__(
        self,
        timeLimit=None,
        iterationLimit=None,
        explorationConstant=1 / math.sqrt(2),
        rolloutPolicy=randomPolicy,
    ):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = "time"
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = "iterations"
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == "time":
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        copy_state = copy.deepcopy(node.state)
        reward = self.rollout(copy_state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        global TEST_NODE
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                copy_state = copy.deepcopy(node.state)
                newNode = treeNode(copy_state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        TEST_NODE = node
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = (
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
            

TEST_NODE = None
game = Game()
m = mcts(timeLimit=1000)
start = time.time()
DEPTH = 6
num_epochs = 4
for e in range(num_epochs):
    num_games = 100
    game_history = []
    for game_counter in range(num_games):
        turn = 0
        board_meta_piece_move = []
        game = Game()
        while not game.is_terminal():
            turn += 1

            if turn == 1:
                game.make_move((4, 4, 1))

            elif turn == 2:
                best_move = random.choice([(3, 3, -1), (4, 3, -1)])
                game.make_move(best_move)

            elif turn % 2:
                if game_counter % 2:
                    best_move = m.search(game)
                    game.make_move(best_move)
                else:
                    flat_board = game.board.flatten()
                    meta = game.meta_available_moves()
                    value, best_move = alpha_beta(
                        game, DEPTH, float("-inf"), float("inf"), True, None
                    )
                    game.make_move(best_move)
                    board_meta_piece_move.append(
                        (flat_board, 1, meta, best_move[0] * 9 + best_move[1])
                    )

            else:
                if not game_counter % 2:
                    best_move = m.search(game)
                    game.make_move(best_move)
                else:
                    flat_board = game.board.flatten()
                    meta = game.meta_available_moves()
                    value, best_move = alpha_beta(
                        game, DEPTH, float("-inf"), float("inf"), False, None
                    )
                    game.make_move(best_move)
                    board_meta_piece_move.append(
                        (flat_board, -1, meta, best_move[0] * 9 + best_move[1])
                    )

        if ((game_counter % 2) and (game.get_winner() == -1)) or (
            (not game_counter % 2) and (game.get_winner() == 1)
        ):
            game_history.append(board_meta_piece_move)
        else:
            print(f"mcts won game {game_counter}")
        print(f"game: {game_counter}, Total elapsed time {time.time() - start}")

    print(f"Finished epoch: {e}")
    with open(f"test_{time.time()}.pickle", "wb+") as f:
        pickle.dump(game_history, f)
print("Finished")