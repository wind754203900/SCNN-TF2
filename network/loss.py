import tensorflow as tf
from tensorflow import keras
from global_config import config
CFG = config.cfg

class LaneExistLoss(keras.losses.Loss):
    """
    Args:
      pos_weight: Scalar to affect the positive labels of the loss function.
      weight: Scalar to affect the entirety of the loss function.
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.sigmoid(y_pred)
        mse = tf.keras.losses.binary_crossentropy(
            y_true, y_pred
        )
        # mse = tf.keras.losses.mean_squared_error(
        #     y_true, y_pred
        # )
        return mse

class WeightedCategoricalCrossentropy(keras.losses.Loss):
    def __init__(self, weights,from_logits=True,alpha=0.8,reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        self.Kweights = tf.constant(weights)
        self.from_logits = from_logits
        self.alpha = alpha
        self.sobel_filter = self._init_sobel_kernel()

    def call(self, y_true, y_pred):
        # CrossEntropy
        ce = tf.losses.categorical_crossentropy(y_true, y_pred, from_logits=self.from_logits) * tf.reduce_sum(
            y_true *self. Kweights, axis=-1)
        ce = tf.reduce_mean(ce)
        return ce
        # l1 loss
        # soomth_l1_loss = self._l1_loss(y_pred)
        # l1 = self.alpha*soomth_l1_loss

        # sobel_loss = self._sobel_loss(y_true,y_pred)
        # sobel_loss = self.alpha*sobel_loss
        #
        # return ce + l1 + sobel_loss
        # return ce + l1

    def _l1_loss(self,y_pred):
        w = y_pred.shape[2]
        diff = tf.abs(y_pred[:, :, 0:w - 2, 1:] - y_pred[:, :, 1:w - 1, 1:])
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
        soomth_l1_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
        soomth_l1_loss = tf.reduce_mean(soomth_l1_loss)
        return soomth_l1_loss

    def _sobel_loss(self,y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred,axis=-1)
        edge_true = tf.nn.conv2d(y_true[:,:,:,:],self.sobel_filter,strides=1,padding='SAME')
        edge_pred = tf.nn.conv2d(y_pred[:, :, :,:], self.sobel_filter, strides=1, padding='SAME')
        diff = tf.abs(edge_true - edge_pred)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
        sobel_loss = (less_than_one * 0.5 * diff ** 2) + (1.0 - less_than_one) * (diff - 0.5)
        sobel_loss = tf.reduce_mean(sobel_loss)
        return sobel_loss

    def _init_sobel_kernel(self):
        sobel_filter = tf.Variable(tf.constant([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]]
        ))
        sobel_filter = sobel_filter[:, :, tf.newaxis, tf.newaxis]
        sobel_filter = tf.tile(sobel_filter, [1, 1, CFG.TRAIN.CLASSES_NUMS - 1, CFG.TRAIN.CLASSES_NUMS - 1])
        return  sobel_filter
