from glob import glob
import tensorflow as tf
import global_config.config as config
import os.path as ops

CFG = config.cfg

def generate_train_dataset(anno_path):
    image_list = []
    mask_list = []
    lane_exist = []
    with open(anno_path, 'r') as file:
        for _info in file:
            info_tmp = _info.strip(' ').split()
            image_list.append(info_tmp[0])
            mask_list.append(info_tmp[1])
            # lane_exist.append([int(info_tmp[2]), int(info_tmp[3]), int(info_tmp[4]), int(info_tmp[5]), int(info_tmp[6])])
            lane_exist.append(list(map(int,info_tmp[2:CFG.TRAIN.CLASSES_NUMS+1])))

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_list,
          {'prob_output':mask_list,
           'exist':lane_exist}))
    dataset = dataset.shuffle(buffer_size=128)
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_data,
                                           batch_size=CFG.TRAIN.BATCH_SIZE,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset,len(image_list) // CFG.TEST.BATCH_SIZE

def generate_test_dataset(anno_path):
    image_list = []
    mask_list = []
    with open(anno_path, 'r') as file:
        for _info in file:
            info_tmp = _info.strip(' ').split()
            image_list.append(info_tmp[0])
            mask_list.append(info_tmp[1])
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_list,
         {'prob_output': mask_list}))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_test_data,
                                           batch_size=CFG.TEST.BATCH_SIZE,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=True))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset,len(image_list) // CFG.TEST.BATCH_SIZE

def get_image(image_path, img_height=720, img_width=1280, mask=False, flip=0,is_aug = False ):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])

        if is_aug:
            img = tf.image.random_brightness(img, max_delta=50.)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
            img = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
            ], default=lambda: img)

        img = tf.clip_by_value(img, 0, 255)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
        if is_aug:
            img = tf.case([
                (tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))
            ], default=lambda: img)
    return img

def random_crop(image, mask, H=512, W=512):
    image_dims = image.shape
    offset_h = tf.random.uniform(
        shape=(1,), maxval=image_dims[0] - H, dtype=tf.int32)[0]
    offset_w = tf.random.uniform(
        shape=(1,), maxval=image_dims[1] - W, dtype=tf.int32)[0]

    image = tf.image.crop_to_bounding_box(image,
                                          offset_height=offset_h,
                                          offset_width=offset_w,
                                          target_height=H,
                                          target_width=W)
    mask = tf.image.crop_to_bounding_box(mask,
                                         offset_height=offset_h,
                                         offset_width=offset_w,
                                         target_height=H,
                                         target_width=W)
    return image, mask

def load_data(image_path, mask_and_exist,
              H=CFG.TRAIN.IMG_HEIGHT, W=CFG.TRAIN.IMG_WIDTH,is_aug=False):
    mask_path, lane_exist = mask_and_exist.get('prob_output'),\
                            mask_and_exist.get('exist')

    flip = tf.random.uniform(
        shape=[1, ], minval=0, maxval=2, dtype=tf.int32)[0]
    image, mask = get_image(image_path, flip=flip,img_height=H,img_width=W,is_aug=is_aug), get_image(
        mask_path, mask=True, flip=flip,img_height=H,img_width=W,is_aug=is_aug)

    if is_aug:
        image, mask = random_crop(image, mask, H=H, W=W)

    # one hot codeing
    mask  = tf.squeeze(tf.one_hot(mask,CFG.TRAIN.CLASSES_NUMS,axis=-1))
    if lane_exist!=None:
        lane_exist = tf.cast(tf.convert_to_tensor(lane_exist),tf.float32)
        return image, (mask, lane_exist)
    else:
        return image, mask

def get_test_image(image_path, img_height=720, img_width=1280, mask=False):
    img = tf.io.read_file(image_path)
    if not mask:
        img = tf.cast(tf.image.decode_png(img, channels=3), dtype=tf.float32)
        img = tf.image.resize(images=img, size=[img_height, img_width])
        img = tf.clip_by_value(img, 0, 255)
        img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    else:
        img = tf.image.decode_png(img, channels=1)
        img = tf.cast(tf.image.resize(images=img, size=[
                      img_height, img_width]), dtype=tf.uint8)
    return img



def load_test_data(image_path, mask_and_exist,
              H=CFG.TRAIN.IMG_HEIGHT, W=CFG.TRAIN.IMG_WIDTH):
    mask_path = mask_and_exist.get('prob_output')
    image, mask = get_test_image(image_path,img_height=H,img_width=W), get_test_image(
        mask_path, mask=True,img_height=H,img_width=W)
    mask = tf.cast(mask,'float32')
    return image, mask

def test_dataset_loader(anno_path,data_root):
    image_list = []
    img_name = []
    with open(anno_path, 'r') as file:
        for _info in file:
            info_tmp = _info.strip(' ').split()
            img_name.append(info_tmp[0])
            image_list.append(ops.join(data_root,info_tmp[0]))
    dataset = tf.data.Dataset.from_tensor_slices((image_list,img_name))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_test_image,
                                           batch_size=CFG.TEST.BATCH_SIZE,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=False))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset, len(image_list) // CFG.TEST.BATCH_SIZE

def load_test_image(image_path,img_name,
              H=CFG.TRAIN.IMG_HEIGHT, W=CFG.TRAIN.IMG_WIDTH):
    image = get_test_image(image_path,img_height=H,img_width=W)
    return image,img_name

def get_culane_test_dataset(data_root,anno_path):
    image_list = []
    img_name = []
    with open(anno_path, 'r') as file:
        for _info in file:
            image_list.append(ops.join(data_root,_info[1:].strip('\n')))
            img_name.append(_info[1:].strip('\n'))
    dataset = tf.data.Dataset.from_tensor_slices((image_list,img_name))
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=load_culane_test_data,
                                           batch_size=CFG.TEST.BATCH_SIZE,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                           drop_remainder=False))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset,len(image_list) // CFG.TEST.BATCH_SIZE

def load_culane_test_data(image_path,img_name):
    image = get_test_image(image_path, img_height=288, img_width=800)
    return image,img_name

if __name__ == '__main__':
    data_root = '/media/wind/MyFile/tusimple_dataset/test_set'
    anno_path = '/media/wind/MyFile/tusimple_dataset/test_set/test.txt'
    dataset = test_dataset_loader(anno_path,data_root)
    print(dataset)
