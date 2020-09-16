import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D, Dropout
from network.base_model import ResNet50
from network.base_model import Vgg16
from network.Module import *
from global_config import config
from network.backbone.resnet18 import ResNet18

CFG = config.cfg

def Upsample(tensor, size,name=None):
    '''bilinear upsampling'''

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size,name=name)(tensor)
    return y


def ASPP(tensor):
    '''atrous spatial pyramid pooling'''
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = Upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y


def DeepLabV3Plus(img_height, img_width, nclasses=6, net_flag = 'resnet50',sub_module=None):
    print('*** Building DeepLabv3Plus Network ***')

    '''
            Building Basic Backbone
    '''
    if net_flag == 'resnet50':
        base_model = ResNet50(input_shape=(
            img_height, img_width, 3), weights='imagenet', include_top=False)
        # get layer feature maps
        image_features = base_model.get_layer('activation_39').output
        x_b = base_model.get_layer('activation_9').output
    elif net_flag== 'vgg16':
        base_model = Vgg16(input_shape=(
            img_height, img_width, 3), weights='imagenet', include_top=False)

        image_features = base_model.get_layer('block5_conv3').output
        x_b = base_model.get_layer('block3_conv3').output
    elif net_flag== 'resnet18':
        base_model = ResNet18(input_shape=(
            img_height, img_width, 3), include_top=False)

        image_features = base_model.get_layer('activation_18').output
        x_b = base_model.get_layer('activation_6').output
    else:
        raise ValueError('net_flag must be \'resnet50\' or \'vgg16\' or \'resnet18\'')
    
    '''
        Deeplabv3
    '''
    x_a = ASPP(image_features)
    x_a = Upsample(tensor=x_a, size=[img_height // 4, img_width // 4])

    x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
    x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
    x_b = Activation('relu', name='low_level_activation')(x_b)

    x = concatenate([x_a, x_b], name='decoder_concat')

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_1')(x)
    x = Activation('relu', name='activation_decoder_1')(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu',
               kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
    x = BatchNormalization(name=f'bn_decoder_2')(x)
    x = Activation('relu', name='activation_decoder_2')(x)

    if sub_module != None:
        if sub_module == 'mp':
            x = MessagePassing(x)
        else:
            raise ValueError('sub_module only support mp now')


    x = Conv2D(nclasses, (1, 1))(Dropout(0.1,name='dropout')(x))
    prob_output = Upsample(x, [img_height, img_width], name='prob_output')


    '''
    output lane existence logits
    shape [N,num_exist]
    '''
    y = tf.nn.softmax(x)
    y = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(y)

    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(128)(y)
    y = tf.keras.layers.ReLU()(y)
    lane_exist = tf.keras.layers.Dense(CFG.TRAIN.CLASSES_NUMS-1,name = 'exist')(y)
    '''
    x = Activation('softmax')(x) 
    tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    Args:
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
    '''
    model = Model(inputs=base_model.input, outputs=[prob_output,lane_exist], name='DeepLabV3_Plus')
    print(f'*** Output_Shape => {model.output_shape} ***')
    return model

if __name__ == '__main__':
    model = DeepLabV3Plus(512, 512, 5,net_flag='resnet50')
    model.summary()
    # x = tf.random.normal([1,512,512,3])
    # out = model(x)
    # print(out.shape)