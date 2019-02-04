import tensorflow as tf


def logit(x):
    return - tf.log(1.0 / x - 1.0)
