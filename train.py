from evaluation_net import evaluationNet
from mct_player import *
from game import *
import config as cfg
import tensorflow as tf
import numpy as np

class Train:
    def __init__(self):
        self.board = Board()
        self.evaluate_net = evaluationNet()
        self.player = MCT_player(self.evaluate_net.evaluate, True)
        self.max_iter = cfg.max_iter
        self.save_file = cfg.save_file
        self.summary_iter = cfg.summary_iter


    def train(self):
        for i in range(self.max_iter):
            actions, probs, value = self.player.next_action_analyze(self.board)
            actions_probs = np.zeros([self.board.board_size * self.board.board_size])
            for i in range(len(actions)):
                actions_probs[actions[i]] = probs[i]
            actions_probs = np.reshape(actions_probs,[-1,self.board.board_size, self.board.board_size])
            state_value = np.array(value)
            state_value = np.reshape(state_value,[-1,1])
            state = self.board.get_state(self.board.current_player)
            state = np.reshape(state,[-1,self.board.board_size,self.board.board_size,3])
            _, loss = self.evaluate_net.session.run([self.evaluate_net.train_op, self.evaluate_net.loss],
                                                    feed_dict={self.evaluate_net.input: state,
                                                               self.evaluate_net.input_mcts_prob: actions_probs,
                                                               self.evaluate_net.mcts_value: state_value})

            if (i + 1) % self.summary_iter == 0:
                print("training iter ",i," loss is ",loss)
                self.evaluate_net.saver.save(self.evaluate_net.session,self.save_file)

            self.player.go_next_action(self.board)
            end, _ = self.board.end_game()
            if end:
                self.player.update_with_move(-1)
                self.board.refresh()
                continue

if __name__ == '__main__':
    training = Train()
    training.train()