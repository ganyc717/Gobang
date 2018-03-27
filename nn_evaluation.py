import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import config as cfg


class NN_evaluation:
    def __init__(self):
        self.board_size = cfg.board_size
        self.input = tf.placeholder(tf.float32,[None, self.board_size, self.board_size, 3])
        #prob and value from mcts search
        self.input_mcts_prob = tf.placeholder(tf.float32,[None, self.board_size, self.board_size])
        self.mcts_value = tf.placeholder(tf.float32,[None, 1])
        self.lr = cfg.learning_rate

        with slim.arg_scope([slim.conv2d,slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input, 32, [3, 3], 1, scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], 1, scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], 1, scope='conv2_2')
            net = slim.conv2d(net, 64, [3, 3], 1, scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], 1, scope='conv3_2')

            action_prob = slim.conv2d(net, 4, [3, 3], 1, scope='action_prob_conv1')
            action_prob = slim.flatten(action_prob, scope='action_prob_flat2')
            action_prob = slim.fully_connected(action_prob,self.board_size * self.board_size,scope = "action_prob_fc3")
            self.action_prob = tf.nn.softmax(action_prob)

            evalutaion = slim.conv2d(net, 8, [3, 3], 1, scope='evalutaion_conv1')
            evalutaion = slim.flatten(evalutaion, scope='evalutaion_flat2')
            evalutaion = slim.fully_connected(evalutaion, 64,scope="evalutaion_fc3")
            self.evalutaion = slim.fully_connected(evalutaion, 1,activation_fn = tf.nn.tanh, scope="evalutaion_fc4")

        # Calculate cross entropy loss
        y = tf.reshape(self.input_mcts_prob,[-1,self.board_size * self.board_size])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(self.action_prob), axis = 1))
        # Calculate value loss
        value_loss = tf.reduce_mean(tf.square(self.evalutaion - self.mcts_value))

        self.loss = value_loss + cross_entropy
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def evaluate(self,board):
        input_state = board.get_state(board.current_player)
        input_state = np.reshape(input_state,[1,self.board_size,self.board_size,3])
        probs,state_value = self.session.run([self.action_prob,self.evalutaion],feed_dict={self.input:input_state})

        probs = np.reshape(probs,[-1])
        action = board.available_move_location
        action_prob = [probs[i] for i in action]

        action_prob = zip(action,action_prob)
        state_value = np.reshape(state_value,[-1])
        return action_prob,state_value

