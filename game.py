import numpy as np
import config as cfg

class Board:
    def __init__(self,board_size = cfg.board_size, win_row_size = cfg.win_row_size):
        self.board_size = board_size
        self.win_row_size = win_row_size
        self.state = np.zeros([self.board_size,self.board_size,2])
        self.available_move_location = range(self.board_size * self.board_size)
        self.player = [0,1]
        self.current_player = 0

    def move(self, location):
        if location not in self.available_move_location:
            print("invalid movement")
            exit(-1)

        row_id = location // self.board_size
        column_id = location % self.board_size

        self.state[row_id,column_id,self.current_player] = 1
        self.available_move_location.remove(location)

        self.current_player = 1 - self.current_player

    def get_state(self,player_id):
        # Get state info from player_id's perspective
        # return state shape is ([board_size,board_size,3])
        # state[:,:,0] is total state, player_id is 1, opponent is -1
        # state[:,:,1] is the location player_id has move
        # state[:,:,2] is the location opponent has move

        assert(player_id in [0,1])
        state = np.zeros([self.board_size,self.board_size,3])
        state[:, :, 1] = self.state[:, :, player_id]
        state[:, :, 2] = self.state[:, :, 1 - player_id]
        state[:, :, 0] = state[:, :, 1] - state[:, :, 2] # state[:, :, 0] = state[:, :, 1] + (-1) * state[:, :, 2]
        return state

    def win_the_game(self,location):
        # return two value
        # game end? win the game?
        row_id = location // self.board_size
        column_id = location % self.board_size
        n = self.win_row_size

        state = self.state[:,:,self.current_player]

        left_bound_limit = min(n - 1, column_id)
        right_bound_limit = min(n - 1, self.board_size - 1 - column_id)

        top_bound_limit = min(n - 1, row_id)
        bottom_bound_limit = min(n - 1, self.board_size - 1 - row_id)

        # horizontal
        for i in range(column_id - left_bound_limit, column_id + right_bound_limit - (n - 1) + 1):
            if np.sum(state[row_id,i:i + n]) > n - 1:
                return True,True
        # vertical
        for i in range(row_id - top_bound_limit, row_id + bottom_bound_limit - (n - 1) + 1):
            if np.sum(state[i:i + n, column_id]) > n - 1:
                return True,True
        # +45 degree
        north_east_limit = min(right_bound_limit, top_bound_limit)
        south_west_limit = min(bottom_bound_limit, left_bound_limit)
        if north_east_limit + south_west_limit < n - 1:
            pass
        else:
            total = row_id + column_id
            for i in range(column_id - south_west_limit, column_id + north_east_limit - (n - 1) + 1):
                sum_ = 0
                for column in range(i, i + n):
                    sum_ += state[total - column, column]
                if sum_ > n - 1:
                    return True,True
        # -45 degree
        north_west_limit = min(left_bound_limit, top_bound_limit)
        south_easy_limit = min(bottom_bound_limit, right_bound_limit)
        if north_west_limit + south_easy_limit < n - 1:
            pass
        else:
            total = column_id -  row_id
            for i in range(row_id - north_west_limit, row_id + south_easy_limit - (n - 1) + 1):
                sum_ = 0
                for row in range(i, i + n):
                    sum_ += state[row, total + row]
                if sum_ > n - 1:
                    return True,True
        if len(self.available_move_location) == 0:
            return True, False
        return False, False