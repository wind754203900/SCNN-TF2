import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

def calculate_AP(label_tensor,input_tensor):
    """
    calculate accuracy acc = correct_nums / ground_truth_nums
    :param input_tensor: binary segmentation logits
    :param label_tensor: binary segmentation label
    :return:
    """

    logits = tf.nn.softmax(input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)
    if K.int_shape(label_tensor)[-1]:
        if K.int_shape(label_tensor)[-1] > 1:
            label_tensor = tf.expand_dims(tf.argmax(label_tensor, axis=-1), axis=-1)

    # binarilize
    final_output = tf.cast(tf.greater(final_output, 0), tf.int32)
    label_tensor = tf.cast(tf.greater(label_tensor, 0), tf.int32)

    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(label_tensor, idx)
    predict_correct_num = tf.math.count_nonzero(pix_cls_ret)
    predict_correct_num = tf.cast(predict_correct_num,'float32')
    gt_correct_num = tf.cast(tf.shape(tf.gather_nd(label_tensor, tf.where(tf.equal(label_tensor, 1))))[0], tf.float32)
    accuracy = tf.divide(
            predict_correct_num+1,
            gt_correct_num+ 1)

    return tf.cast(accuracy,tf.float32)

def calculate_IoU(label_tensor,input_tensor):
    logits = tf.nn.softmax(input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)
    if K.int_shape(label_tensor)[-1]:
        if K.int_shape(label_tensor)[-1] > 1 :
            label_tensor = tf.expand_dims(tf.argmax(label_tensor, axis=-1), axis=-1)

    # binarilize
    final_output = tf.cast(tf.greater(final_output, 0), tf.int32)
    label_tensor = tf.cast(tf.greater(label_tensor, 0), tf.int32)

    # cal IoU
    overlap_1 = tf.math.count_nonzero(tf.multiply(tf.cast(tf.equal(label_tensor, 1), tf.int32),
                                                   tf.cast(tf.equal(final_output, 1), tf.int32)),
                                       dtype=tf.int32)
    union_1 = tf.add(tf.math.count_nonzero(tf.cast(tf.equal(label_tensor, 1),
                                                       tf.int32), dtype=tf.int32),
                         tf.math.count_nonzero(tf.cast(tf.equal(final_output, 1),
                                                       tf.int32), dtype=tf.int32))
    union_1 = tf.subtract(union_1, overlap_1)
    IoU = tf.divide(tf.cast(overlap_1, tf.float32)+1, (tf.cast(union_1, tf.float32) + 1))
    IoU = tf.reduce_mean(IoU)
    return IoU

def calculate_model_fp(label_tensor,input_tensor):
    """
    calculate fp figure
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)
    if K.int_shape(label_tensor)[-1]:
        if K.int_shape(label_tensor)[-1] > 1:
            label_tensor = tf.expand_dims(tf.argmax(label_tensor, axis=-1), axis=-1)

    final_output = tf.cast(tf.greater(final_output, 0), tf.int32)
    label_tensor = tf.cast(tf.greater(label_tensor, 0), tf.int32)

    idx = tf.where(tf.equal(final_output, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    false_pred = tf.cast(tf.shape(pix_cls_ret)[0], tf.int64) - tf.math.count_nonzero(
        tf.gather_nd(label_tensor, idx)
    )
    fp = tf.divide(false_pred, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    return tf.cast(fp,tf.float32)

def calculate_model_fn(label_tensor,input_tensor):
    """
    calculate fn figure
    :param input_tensor:
    :param label_tensor:
    :return:
    """
    logits = tf.nn.softmax(input_tensor)
    final_output = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)
    if K.int_shape(label_tensor)[-1]:
        if K.int_shape(label_tensor)[-1] > 1:
            label_tensor = tf.expand_dims(tf.argmax(label_tensor, axis=-1), axis=-1)

    final_output = tf.cast(tf.greater(final_output, 0), tf.int32)
    label_tensor = tf.cast(tf.greater(label_tensor, 0), tf.int32)

    idx = tf.where(tf.equal(label_tensor, 1))
    pix_cls_ret = tf.gather_nd(final_output, idx)
    label_cls_ret = tf.gather_nd(label_tensor, tf.where(tf.equal(label_tensor, 1)))
    mis_pred = tf.cast(tf.shape(label_cls_ret)[0], tf.int64) - tf.math.count_nonzero(pix_cls_ret)

    fn = tf.divide(mis_pred, tf.cast(tf.shape(label_cls_ret)[0], tf.int64))

    return tf.cast(fn,tf.float32)