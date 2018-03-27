import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg

class NN_evaluation:
    def __init__(self):
        self.board_size = cfg.board_size
        self.input = tf.placeholder(tf.float32,[None, self.board_size, self.board_size, 3])
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(self.input, 32, [3, 3], 1, scope='conv1')
            net = slim.conv2d(net, 32, [3, 3], 1, scope='conv2_1')
            net = slim.conv2d(net, 64, [3, 3], 1, scope='conv2_2')
            net = slim.conv2d(net, 64, [3, 3], 1, scope='conv3_1')
            net = slim.conv2d(net, 128, [3, 3], 1, scope='conv3_2')

            action_prob = slim.conv2d(net, 4, [3, 3], 1, scope='action_prob_conv1')
            action_prob = slim.flatten(action_prob, scope='action_prob_flat2')
            action_prob = slim.fully_connected(action_prob,self.board_size * self.board_size,weights_regularizer=slim.l2_regularizer(0.0005),scope = "action_prob_fc3")
            self.action_prob = tf.nn.softmax(action_prob)

            evalutaion = slim.conv2d(net, 8, [3, 3], 1, scope='evalutaion_conv1')
            evalutaion = slim.flatten(evalutaion, scope='evalutaion_flat2')
            evalutaion = slim.fully_connected(evalutaion, 64,weights_regularizer=slim.l2_regularizer(0.0005),scope="evalutaion_fc3")
            self.evalutaion = slim.fully_connected(evalutaion, 1,activation_fn = tf.nn.tanh,weights_regularizer=slim.l2_regularizer(0.0005), scope="evalutaion_fc4")