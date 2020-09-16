import tensorflow as tf
import tensorflow.keras.backend as K
from global_config import config
import cv2
import numpy as np
import os

#   CFG.TEST.IMG_SAVE_DIR : the path you want the visualize image store

CFG = config.cfg

id_to_color = {
    1:(0, 255, 0),
    2:(227, 27, 13),
    3:(227,255,13),
    4:(111,74,0),
    5:(153,153,153)
}

def write_img(src_img, pred_tensor, step=None, exist=None):
    pred_tensor = tf.nn.softmax(pred_tensor)
    final_output = tf.expand_dims(tf.argmax(pred_tensor, axis=-1), axis=-1)

    #   convert instance image to binary image
    final_output = tf.cast(tf.greater(final_output, 0), tf.int32)

    batch_size = K.int_shape(src_img)[0]
    # de-nomoralize source image
    src_img = src_img[:,:,:,::-1] + tf.constant([103.939, 116.779, 123.68])
    alpha = 0.5
    for i in range(batch_size):
        img_color = src_img[i].numpy()
        pred_img = final_output[i, :, :, 0].numpy()
        index = step * batch_size + i

        for j in np.unique(final_output[i]):
            if j in id_to_color:
                img_color[pred_img == j] = id_to_color[j]

        exist_list = [1 if j >= 0.5 else 0 for j in exist.numpy()[i]]
        exist_result = "".join(list(map(str, exist_list)))

        # check dir
        save_dir = CFG.TEST.IMG_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        name = os.path.join(save_dir,'seg_{}_{}.png'.format(index,exist_result))
        cv2.addWeighted(src_img[i].numpy(), alpha, img_color, 1 - alpha, 0, img_color)

        predict_image = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)

        cv2.imwrite(name,predict_image)
