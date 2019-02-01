import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, necessary for 3d projection
import tensorflow as tf


class Model:

    def __init__(self, env, alpha=0.001, discount=0.9):

        self.env = env
        self.alpha = alpha
        self.discount = discount

        self.features_t = None
        self.reward_t = None
        self.successor_t = None
        self.successor_pi_t = None
        self.reward_loss_t = None
        self.successor_loss_t = None
        self.loss_t = None
        self.train_op = None
        self.session = None

        self.build_model()
        self.build_training()

    def build_model(self):

        self.features_t = tf.get_variable(
            "features", shape=(self.env.NUM_STATES, self.env.NUM_FEATURES),
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        self.rewards_t = tf.get_variable(
            "rewards", shape=(self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        self.successor_t = tf.get_variable(
            "successor", shape=(self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        self.successor_pi_t = tf.reduce_mean(self.successor_t, axis=1)

    def build_training(self):

        self.reward_loss_t = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(tf.matmul(self.features_t, self.rewards_t, name="o1") - self.env.r), axis=0
            ), axis=0
        )

        middle1 = tf.stack(
            [tf.matmul(self.env.p[:, :, i], self.features_t) for i in range(self.env.NUM_ACTIONS)], axis=2
        )

        middle2 = tf.stack(
            [tf.matmul(middle1[:, :, i], self.successor_pi_t[:, tf.newaxis])[:, 0]
             for i in range(self.env.NUM_ACTIONS)], axis=1
        )

        self.successor_loss_t = tf.reduce_mean(
            [tf.reduce_sum(
                self.features_t + (self.discount * middle2)[:, i: i + 1] -
                tf.matmul(self.features_t, self.successor_t)[:, i: i + 1]
            ) for i in range(self.env.NUM_ACTIONS)]
        )

        self.loss_t = self.reward_loss_t + self.alpha * self.successor_loss_t

        self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss_t)

    def train_step(self):

        loss, _ = self.session.run([self.loss_t, self.train_op])
        return loss

    def show_feature_space(self):

        features = self.session.run(self.features_t)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features[:, 0], features[:, 1], features[:, 2])

        plt.show()

    def start_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()
