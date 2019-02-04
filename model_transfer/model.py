import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # don't delete this, necessary for 3d projection
from sklearn.cluster import KMeans
import tensorflow as tf
from . import utils


class Model:

    def __init__(self, env, alpha=0.001, discount=0.9, learning_rate=0.001):

        self.env = env
        self.alpha = alpha
        self.discount = discount
        self.learning_rate = learning_rate

        self.features_t = None
        self.features_squashed_t = None
        self.rewards_t = None
        self.successor_t = None
        self.new_features_pl = None
        self.new_rewards_pl = None
        self.new_successor_pl = None
        self.assign_features_op = None
        self.assign_rewards_op = None
        self.assign_successor_op = None
        self.successor_pi_t = None
        self.reward_loss_t = None
        self.successor_loss_t = None
        self.loss_t = None
        self.train_op = None
        self.session = None

        self.build_model()
        self.build_training()

    def train_step(self):

        loss, _ = self.session.run([self.loss_t, self.train_op])
        return loss

    def show_feature_space(self):

        features = self.session.run(self.features_squashed_t)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=[1, 2, 3] * 30)

        plt.show()

    def k_means_update(self):

        features, rewards, successor = self.session.run([self.features_t, self.rewards_t, self.successor_t])

        k_means = KMeans(n_clusters=self.env.NUM_FEATURES, random_state=0)
        k_means.fit(features)

        m = k_means.cluster_centers_.transpose((1, 0))

        m_inverse = np.linalg.inv(m)

        new_features = np.matmul(features, m_inverse)
        new_rewards = np.matmul(m, rewards)

        new_successor = np.stack(
            [np.matmul(np.matmul(m, successor[:, :, i]), m_inverse) for i in range(self.env.NUM_ACTIONS)],
            axis=-1
        )

        self.session.run([self.assign_features_op, self.assign_rewards_op, self.assign_successor_op], feed_dict={
            self.new_features_pl: new_features,
            self.new_rewards_pl: new_rewards,
            self.new_successor_pl: new_successor
        })

    def policy_evaluation(self, policy, values, threshold=0.001):

        features, rewards, successor, successor_pi = self.session.run(
            [self.features_squashed_t, self.rewards_t, self.successor_t, self.successor_pi_t]
        )

        transitions = np.stack(
            [np.matmul((successor[:, :, i] - np.identity(len(successor))),
                       np.linalg.inv(successor_pi)) / self.discount for i in range(self.env.NUM_ACTIONS)],
            axis=-1)
        print(transitions)
        diag_policy = np.stack([np.diag(policy[:, a]) for a in range(policy.shape[1])], axis=-1)

        prev_v = None
        v = np.zeros(self.env.NUM_FEATURES, dtype=np.float32)

        while prev_v is None or np.max(np.abs(prev_v - v)) > threshold:

            prev_v = v

            q = np.stack(
                [rewards[:, i] + self.discount * np.matmul(transitions[:, :, i], v)
                 for i in range(self.env.NUM_ACTIONS)],
                axis=-1
            )

            v = np.matmul(np.linalg.pinv(features), np.sum(
                [np.matmul(np.matmul(diag_policy[:, :, i], features), q[:, i]) for i in range(self.env.NUM_ACTIONS)],
                axis=0
            ))

        diff = np.max(np.abs(np.matmul(features, v) - values))

        return diff

    def build_model(self):

        self.features_t = tf.get_variable(
            "features", shape=(self.env.NUM_STATES, self.env.NUM_FEATURES),
            initializer=tf.random_normal_initializer(mean=0.0, stddev=2.0, dtype=tf.float32)
        )
        self.features_squashed_t = tf.nn.sigmoid(self.features_t)

        self.rewards_t = tf.get_variable(
            "rewards", shape=(self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        self.successor_t = tf.get_variable(
            "successor", shape=(self.env.NUM_FEATURES, self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
            initializer=tf.random_uniform_initializer(minval=0, maxval=1, dtype=tf.float32)
        )

        self.new_features_pl = tf.placeholder(tf.float32, shape=(self.env.NUM_STATES, self.env.NUM_FEATURES),
                                              name="new_features_pl")
        self.new_rewards_pl = tf.placeholder(tf.float32, shape=(self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
                                             name="new_rewards_pl")
        self.new_successor_pl = tf.placeholder(
            tf.float32, shape=(self.env.NUM_FEATURES, self.env.NUM_FEATURES, self.env.NUM_ACTIONS),
            name="new_successor_pl"
        )

        self.assign_features_op = tf.assign(self.features_t, self.new_features_pl)
        self.assign_rewards_op = tf.assign(self.rewards_t, self.new_rewards_pl)
        self.assign_successor_op = tf.assign(self.successor_t, self.new_successor_pl)

        self.successor_pi_t = tf.reduce_mean(self.successor_t, axis=-1)

    def build_training(self):

        self.reward_loss_t = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(tf.matmul(self.features_squashed_t, self.rewards_t, name="o1") - self.env.r), axis=0
            ), axis=0
        )

        # states x features x 1
        term1 = self.features_squashed_t[:, :, tf.newaxis]

        # states x feature x actions
        term2 = tf.stack(
            [tf.matmul(self.env.p[:, :, i], self.features_squashed_t) for i in range(self.env.NUM_ACTIONS)], axis=-1
        )

        term2 = tf.stack(
            [tf.matmul(term2[:, :, i], self.successor_pi_t)
             for i in range(self.env.NUM_ACTIONS)], axis=-1
        )

        # states x features x actions
        term3 = tf.stack(
            [tf.matmul(self.features_squashed_t, self.successor_t[:, :, i]) for i in range(self.env.NUM_ACTIONS)],
            axis=-1
        )

        assert term2.shape == term3.shape
        assert len(term1.shape) == len(term2.shape)

        self.successor_loss_t = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(term1 + self.discount * term2 - term3), axis=[0, 1]
            ), axis=0
        )

        self.loss_t = self.reward_loss_t + self.alpha * self.successor_loss_t

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_t)

    def start_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()
