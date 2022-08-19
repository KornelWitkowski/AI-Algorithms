import random
import tensorflow as tf
from tensorflow.keras import Sequential, Input, regularizers
from tensorflow.keras.layers import Dense
from random import choice
from copy import deepcopy
import numpy as np

BLANK = 0

PLAYER_1_MARK = -1
PLAYER_2_MARK = 1

REWARD_WIN = 10
REWARD_LOSE = -100
REWARD_DRAW = 1

MARKS = {-1: 'X', 0: ' ', 1: 'O'}
ROTATION = {0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7, 6: 0, 7: 3, 8: 6}


def rotate_board(board, angle=1, backward=False):
    if backward:
        if angle == 1:
            angle = 3
        elif angle == 3:
            angle = 1

    if angle == 0:
        return board
    if angle > 1:
        board = rotate_board(board, angle-1)

    new_board = [""] * 9

    for i in range(9):
        new_board[i] = board[ROTATION[i]]

    return new_board


def rotate_move(move, angle=1):
    return rotate_board([0, 1, 2, 3, 4, 5, 6, 7, 8], angle, backward=True)[move]


def create_model():
    model = Sequential([Input(shape=(10,)),
                        Dense(32, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)),
                        Dense(32, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4)),
                        Dense(1, activation='linear')])
    model.compile(loss='mse', optimizer='adam')
    return model


class Player:

    @staticmethod
    def show_board(board):
        board = [MARKS[square] for square in board]
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))

    @staticmethod
    def available_squares(board):
        return [k for k in range(9) if board[k] == BLANK]


class HumanPlayer(Player):

    def reward(self, value, board, moves):
        pass

    def make_move(self, board):

        while True:
            try:
                self.show_board(board)
                move = input('Your next move (cell index 1-9):')
                move = int(move)

                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move-1


class AIPlayer(Player):

    def __init__(self, mark, model, epsilon=0.4, gamma=0.9):
        # the probability of exploration
        self.mark = mark
        self.epsilon = epsilon

        self.GAMMA = gamma
        self.q_model = model

        # previous move during the game
        self.move = None
        self.board = (BLANK,) * 9

        self.boards = []
        self.moves = []

    def fit_q_model(self, x, y, epochs=1):
        self.q_model.fit(x, y, epochs=epochs, verbose=0)

    def get_q_model(self, state, action):
        # the input must be reshape to a 'batch shape'
        return self.q_model(np.array(list(state) + [action]).reshape(1, 10), training=False).numpy()[0][0]

    def make_move(self, board):

        self.board = tuple(board)
        actions = self.available_squares(board)

        # exploration vs exploitation: with the probability epsilon it makes random move
        if random.random() < self.epsilon:
            self.move = random.choice(actions)
            return self.move

        q_values = [self.get_q_model(self.board, a) for a in actions]
        max_q_value = tf.reduce_max(q_values).numpy()
        best_actions = [k for k in range(len(actions)) if q_values[k] == max_q_value]

        best_move = actions[choice(best_actions)]
        self.move = best_move

        return self.move

    def reward(self, reward, boards, moves):

        if self.move:
            x, y = [], []

            # win or draw
            if boards[-2][1] == self.mark:
                for i in range(4):
                    prev_q = self.get_q_model(rotate_board(boards[-2][0], i), rotate_move(moves[-1], i))
                    x.append(list(rotate_board(boards[-2][0], i)) + [rotate_move(moves[-1], i)])
                    y.append(prev_q + reward)

            # lose or draw
            if boards[-2][1] == -self.mark:
                for i in range(4):
                    prev_q = self.get_q_model(rotate_board(boards[-3][0], i), rotate_move(moves[-2], i))
                    x.append(list(rotate_board(boards[-3][0], i)) + [rotate_move(moves[-2], i)])
                    y.append(prev_q + reward)

            self.fit_q_model(x, y, 3)


class TicTacToe:

    def __init__(self, player1, player2, verbose=1):
        self.player1 = player1
        self.player2 = player2
        self.verbose = verbose
        self.first_player_turn = random.choice([True, False])
        self.board = [BLANK] * 9

        # in the below list a history of the game is stored for further training
        action = PLAYER_1_MARK if self.first_player_turn == 1 else PLAYER_2_MARK
        self.boards = [(deepcopy(self.board), action)]
        self.moves = []

    def play(self):

        while True:
            if self.first_player_turn:
                player, other_player = self.player1, self.player2
                player_marks = (PLAYER_1_MARK, PLAYER_2_MARK)
            else:
                player, other_player = self.player2, self.player1
                player_marks = (PLAYER_2_MARK, PLAYER_1_MARK)

            game_over, winner = self.is_game_over(player_marks)

            if game_over:

                if not winner:
                    result = 'Draw!'
                    player.reward(REWARD_DRAW, self.boards, self.moves)
                    other_player.reward(REWARD_DRAW, self.boards, self.moves)
                else:
                    result = f'{other_player.__class__.__name__} won!'
                    other_player.reward(REWARD_WIN, self.boards, self.moves)
                    player.reward(REWARD_LOSE, self.boards, self.moves)

                if self.verbose:
                    player.show_board(self.board[:])
                    print(f"\n{result}\n")

                break

            self.first_player_turn = not self.first_player_turn

            # actual player makes move
            move = player.make_move(self.board)
            self.board[move] = player_marks[0]

            # here we store the history
            self.moves.append(move)
            self.boards.append((deepcopy(self.board), player_marks[1]))

        return winner

    def is_game_over(self, player_marks):

        for player_mark in player_marks:

            # check rows
            for i in range(3):
                if self.board[3 * i + 0] == player_mark and\
                        self.board[3 * i + 1] == player_mark and\
                        self.board[3 * i + 2] == player_mark:
                    return True, player_mark

            # check columns
            for j in range(3):
                if self.board[j + 0] == player_mark and \
                        self.board[j + 3] == player_mark and \
                        self.board[j + 6] == player_mark:
                    return True, player_mark

            # check diagonals
            if self.board[0] == player_mark and self.board[4] == player_mark and\
                    self.board[8] == player_mark:
                return True, player_mark

            if self.board[2] == player_mark and self.board[4] == player_mark and self.board[6] == player_mark:
                return True, player_mark

        # check if draw
        if self.board.count(BLANK) == 0:
            return True, 0
        else:
            return False, None


if __name__ == '__main__':
    tic_tac_toe_model = create_model()
    ai_player_1 = AIPlayer(PLAYER_1_MARK, tic_tac_toe_model)
    ai_player_2 = AIPlayer(PLAYER_2_MARK, tic_tac_toe_model)

    print('Training the AI player...')

    ai_player_1.epsilon = 0     # full exploitation
    ai_player_2.epsilon = 1     # full exploration
    # the AI players share the model, so exploration and exploitation is used to train the same network

    j = 0

    for n in range(1, 200+1):
        game = TicTacToe(ai_player_1, ai_player_2, verbose=0)
        j += game.play()

        if n % 200 == 0:
            print(f"Iteration: {n}")
            print(f"Score: {-j}")
            j = 0

    print('\nTraining is done')

    human_player = HumanPlayer()
    game = TicTacToe(ai_player_1, human_player)
    game.play()
