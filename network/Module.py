import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import math

def MessagePassing(tensor):
    dims = K.int_shape(tensor)

    '''
    # top to down #
    '''
    feature_list_old = []
    feature_list_new = []
    for cnt in range(dims[1]):
        feature_list_old.append(tf.expand_dims(tensor[:, cnt, :, :], axis=1))
    feature_list_new.append(tf.expand_dims(tensor[:, 0, :, :], axis=1))

    t2d_layer = layers.Conv2D(dims[-1], kernel_size=[1,9],strides=(1,1),
                                     kernel_initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))),
                                     padding='SAME',
                                     name='t2d_conv')
    relu = layers.Activation('relu',name='t2d_relu')
    feature = t2d_layer(feature_list_old[0])
    feature = relu(feature) + feature_list_old[1]
    feature_list_new.append(feature)

    for cnt in range(2, dims[1]):
        feature = t2d_layer(feature_list_new[cnt - 1])
        feature = relu(feature) + feature_list_old[cnt]
        feature_list_new.append(feature)

    '''
    # down to top #
    '''
    feature_list_old = feature_list_new
    feature_list_new = []
    length = dims[1] - 1
    feature_list_new.append(feature_list_old[length])

    d2t_layer = layers.Conv2D(dims[-1], kernel_size=[1, 9], strides=(1, 1),
                                   kernel_initializer=tf.random_normal_initializer(0, math.sqrt(
                                       2.0 / (9 * 128 * 128 * 5))),
                                   padding='SAME',
                                   name='d2t_layer_conv')
    relu = layers.Activation('relu', name='d2t_relu')

    feature = d2t_layer(feature_list_old[length])
    feature = relu(feature) + feature_list_old[length - 1]
    feature_list_new.append(feature)

    for cnt in range(2, dims[1]):
        feature = d2t_layer(feature_list_new[cnt - 1])
        feature = relu(feature) + feature_list_old[length - cnt]
        feature_list_new.append(feature)

    feature_list_new.reverse()

    '''
    # stack feature #
    '''
    processed_feature = tf.stack(feature_list_new, axis=1)
    processed_feature = tf.squeeze(processed_feature, axis=2)

    '''
    # left to right #
    '''
    feature_list_old = []
    feature_list_new = []
    for cnt in range(processed_feature.shape[2]):
        feature_list_old.append(tf.expand_dims(processed_feature[:, :, cnt, :], axis=2))
    feature_list_new.append(tf.expand_dims(processed_feature[:, :, 0, :], axis=2))

    l2r_layer = layers.Conv2D(dims[-1], kernel_size=[9, 1], strides=(1, 1),
                                     kernel_initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))),
                                     padding='SAME',
                                     name='l2r_conv')
    relu = layers.Activation('relu', name='l2r_relu')

    feature = l2r_layer(feature_list_old[0])
    feature = relu(feature) + feature_list_old[1]
    feature_list_new.append(feature)

    for cnt in range(2, processed_feature.shape[2]):
        feature = l2r_layer(feature_list_new[cnt - 1])
        feature = relu(feature) + feature_list_old[cnt]
        feature_list_new.append(feature)

    '''
    # right to left #
    '''

    r2l_layer = layers.Conv2D(dims[-1], kernel_size=[9, 1], strides=(1, 1),
                  kernel_initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))),
                  padding='SAME',
                  name='r2l_conv')
    relu = layers.Activation('relu', name='r2l_relu')

    feature_list_old = feature_list_new
    feature_list_new = []
    length = dims[2] - 1
    feature_list_new.append(feature_list_old[length])

    feature = r2l_layer(feature_list_old[length])
    feature = relu(feature) + feature_list_old[length - 1]
    feature_list_new.append(feature)

    for cnt in range(2, processed_feature.shape[2]):
        feature = r2l_layer(feature_list_new[cnt - 1])
        feature = relu(feature) + feature_list_old[length - cnt]
        feature_list_new.append(feature)

    '''
    # stack feature #
    '''
    feature_list_new.reverse()
    processed_feature = tf.stack(feature_list_new, axis=2)
    processed_feature = tf.squeeze(processed_feature, axis=3)

    return processed_feature