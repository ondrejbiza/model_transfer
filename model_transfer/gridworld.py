import numpy as np


class GridWorld:

    NUM_STATES = 90
    NUM_FEATURES = 3
    NUM_ACTIONS = 4

    def __init__(self):

        # initialize empty transition and reward tables
        self.p = np.zeros((30, 3, 30, 3, 4), dtype=np.float32)
        self.r = np.zeros((30, 3, 4), dtype=np.float32)

        # fill in the tables
        # order of actions: up, down, left, right
        self.r[:, 2, 0] = 1
        self.r[:, 2, 1] = 1
        self.r[:, 2, 3] = 1
        self.r[:, 1, 3] = 1

        # transitions for up
        for column in range(3):
            self.p[0, column, 0, column, 0] = 1
            for row in range(1, 30):
                self.p[row, column, row - 1, column, 0] = 1

        # transitions for down
        for column in range(3):
            self.p[29, column, 29, column, 1] = 1
            for row in range(0, 29):
                self.p[row, column, row + 1, column, 1] = 1

        # transitions for left
        for column in range(3):
            for row in range(0, 30):
                if column == 0:
                    self.p[row, column, row, column, 2] = 1
                else:
                    self.p[row, column, row, column - 1, 2] = 1

        # transitions for right
        for column in range(3):
            for row in range(0, 30):
                if column == 2:
                    self.p[row, column, row, column, 3] = 1
                else:
                    self.p[row, column, row, column + 1, 3] = 1

        # resize the matrices
        self.p = np.reshape(self.p, (90, 90, 4))
        self.r = np.reshape(self.r, (90, 4))

        # policies
        self.uniform_policy = np.zeros((90, 4), dtype=np.float32) + 0.25
        self.uniform_policy_values = self.value_iteration(self.uniform_policy)

        self.optimal_policy = np.zeros((90, 4), dtype=np.float32)
        self.optimal_policy[:, 3] = 1.0
        self.optimal_policy_values = self.value_iteration(self.optimal_policy)

        self.half_optimal_policy = np.zeros((90, 4), dtype=np.float32)
        self.half_optimal_policy[:, 3] = 0.5
        self.half_optimal_policy[:, :3] = 0.5 / 3
        self.half_optimal_policy_values = self.value_iteration(self.half_optimal_policy)

    def value_iteration(self, policy, discount=0.9, threshold=1-6, max_steps=1+6):

        max_diff = threshold + 1
        step = 0
        q = np.zeros((90, 4), dtype=np.float32)
        v = None

        while max_diff > threshold and step < max_steps:

            # get state values under the current policy
            v = np.sum(policy * q, axis=1)

            # get new q-values
            new_q = self.r + discount * np.sum(self.p * v[np.newaxis, :, np.newaxis], axis=1)

            # get the maximum different between the old and new q-values
            max_diff = np.max(np.abs(q - new_q))

            # save q-values
            q = new_q

            step += 1

        return v
