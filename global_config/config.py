from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Dataset Path
__C.TU_DATASETS_TRAIN = '{your_generated_tusimple_dataset_path}/training/train_instance.txt'

__C.TU_DATASETS_VALID = '{your_generated_tusimple_dataset_path}/training/validation_instance.txt'

# Image Original Size
__C.ORI_IMG_HEIGHT = 720
__C.ORI_IMG_WIDTH = 1280

# Train options
__C.TRAIN = edict()

'''
    Training setting
'''
# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 100
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 4e-4
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 8# 4
# Set the shadownet validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 8  # 4
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 3000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.8
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 5
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256  # 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 512  # 512

# Set the backbone ('vgg16' or 'resnet18' or 'resnet50')
__C.TRAIN.FLAG = 'resnet18'
# Set if using Message Passing Module
__C.TRAIN.SUB_MODULE = 'mp' # mp or None,if mp: use message passing module
# Set the weights save path
__C.TRAIN.MODEL_SAVE_PATH = '../weights/{}_deeplab_{}.h5'.format(__C.TRAIN.FLAG,__C.TRAIN.SUB_MODULE) if __C.TRAIN.SUB_MODULE !=None \
    else '../weights/{}_deeplab.h5'.format(__C.TRAIN.FLAG)



# Test options
__C.TEST = edict()
'''
    Test setting
'''
# Set the test batch size
__C.TEST.BATCH_SIZE = 8
# Set the backbone ('vgg16' or 'resnet18' or 'resnet50')
__C.TEST.FLAG = 'resnet18'
# Set the store path of test image
__C.TEST.IMG_SAVE_DIR  = '../test_result/{}/'.format(__C.TEST.FLAG)

# Set if using Message Passing Module
__C.TEST.SUB_MODULE = 'mp' # mp or None,if mp: use message passing module

# Set the restore weights path
__C.TEST.MODEL_LOAD_PATH = '../weights/{}_deeplab_{}.h5'.format(__C.TEST.FLAG,__C.TEST.SUB_MODULE) if __C.TEST.SUB_MODULE !=None \
    else '../weights/{}_deeplab.h5'.format(__C.TEST.FLAG)

__C.TEST.DATAROOT = '{your_tusimple_test_dataset_path}/test_set'
__C.TEST.ANNO_PATH = '{your_tusimple_test_dataset_path}/test_set/test.txt'

# Set if visualize result
__C.TEST.VISUALIZE = False