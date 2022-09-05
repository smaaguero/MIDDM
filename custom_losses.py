import tensorflow as tf
from tensorflow.keras import backend as K

def weighted_binary_crossentropy(hyperparameters):
    w1 = hyperparameters["w1"]
    w2 = hyperparameters["w2"]
    """
    Binary form of weighted binary cross entropy.
      WBCE(p_t) = -w * (1 - p_t)* log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    Usage:
     model.compile(loss=[weighted_binary_crossentropyv2(hyperparameters)], metrics=["accuracy"], optimizer=adam)
    """
    def loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(w1 * (1. - pt_1) * K.log(pt_1)) \
               -K.sum(w2 * (pt_0) * K.log(1. - pt_0))

    return loss