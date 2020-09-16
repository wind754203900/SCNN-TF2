from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from network.deeplab import DeepLabV3Plus
from data_provider.dataset_maker import *
from network.loss import LaneExistLoss, WeightedCategoricalCrossentropy
from network.metrics import calculate_AP,calculate_IoU
import global_config.config as config

CFG = config.cfg

print('TensorFlow', tf.__version__)

import sys
sys.setrecursionlimit(9000000)

train_path = CFG.TU_DATASETS_TRAIN
val_path = CFG.TU_DATASETS_VALID
classes_nums = CFG.TRAIN.CLASSES_NUMS

train_dataset,train_data_len = generate_train_dataset(train_path)
val_dataset,val_data_len = generate_train_dataset(val_path)

print(train_dataset)

weight = [1.0 if i!=0 else 0.4 for i in range(classes_nums) ]

prob_out_loss = WeightedCategoricalCrossentropy(weights=weight,from_logits=True)
exist_loss = LaneExistLoss()
strategy = tf.distribute.MirroredStrategy()

H,W = CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH

with strategy.scope():
    model = DeepLabV3Plus(H, W, classes_nums,
                          net_flag=CFG.TRAIN.FLAG, sub_module=CFG.TRAIN.SUB_MODULE)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=CFG.TRAIN.LEARNING_RATE,
        decay_steps=CFG.TRAIN.LR_DECAY_STEPS,
        decay_rate=CFG.TRAIN.LR_DECAY_RATE,
        staircase=True
    )
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

    model.compile(loss={'prob_output':prob_out_loss,
                        'exist':exist_loss},
                  loss_weights={'prob_output': 1., 'exist': 0.01},
                  optimizer=optimizer,
                  metrics={'prob_output':[calculate_AP,calculate_IoU]})


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='auto', filepath=CFG.TRAIN.MODEL_SAVE_PATH,
                     save_best_only='True',
                     save_weights_only='True',verbose=1)
es = EarlyStopping(patience=5,min_delta=0.0001)

callbacks = [mc, tb, es]

# model.summary()

model.fit(train_dataset,
          steps_per_epoch=train_data_len,
          epochs=CFG.TRAIN.EPOCHS,
          validation_data=val_dataset,
          validation_steps=val_data_len,
          callbacks=callbacks
          )
