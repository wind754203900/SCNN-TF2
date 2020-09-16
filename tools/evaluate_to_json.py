from network.deeplab import DeepLabV3Plus
from data_provider.dataset_maker import *
from network.loss import *
from global_config import config
from utils.postprocess import Postprocess
from utils.lane import LaneEval
from utils.visualize import write_img
import json

import sys
sys.setrecursionlimit(9000000)


'''
    grount truth downlaod url: https://github.com/TuSimple/tusimple-benchmark/issues/3
'''

save_name = 'evaluate_lane.json'    #   evalution result will be put in ../evaluation/{save_name}
gt_json_path = '/media/wind/MyFile/tusimple_dataset/test_set/test_label.json'     # modify to your ground truth test label json file path


CFG = config.cfg
# prepare data
anno_path,data_root = CFG.TEST.ANNO_PATH,CFG.TEST.DATAROOT
test_data,data_len = test_dataset_loader(anno_path,data_root)

print(test_data)

# prepare model
H,W,classes_nums = CFG.TRAIN.IMG_HEIGHT,CFG.TRAIN.IMG_WIDTH,CFG.TRAIN.CLASSES_NUMS
model = DeepLabV3Plus(H, W, classes_nums,
                          net_flag=CFG.TEST.FLAG, sub_module=CFG.TEST.SUB_MODULE)

# load weights
model.load_weights(CFG.TEST.MODEL_LOAD_PATH)

# instance Postprocess
postprocess = Postprocess()

with tf.device('/gpu:0'),open('../evaluation/{}'.format(save_name),'w') as file:
    for i,(img,name) in enumerate(test_data.as_numpy_iterator()):
        pred,exist = model(img,training=False)
        pred = tf.convert_to_tensor(pred)
        pred = tf.nn.softmax(pred)

        '''
            visulize to image
        '''
        if CFG.TEST.VISUALIZE:
            exist = tf.nn.sigmoid(exist)
            write_img(img, pred, step=i, exist=exist)

        '''
            convert result to json file
        '''
        for idx in range(pred.shape[0]):
            # pred[idx]: index of batch size,denotes one image
            binary_img = pred[idx]
            lane_exist = exist[idx]

            # process instance image to binary image
            binary_img = tf.argmax(binary_img, axis=-1)
            binary_img = tf.one_hot(binary_img,classes_nums)
            binary_img = binary_img[:,:,1:classes_nums]
            binary_img = tf.image.resize(binary_img,[CFG.ORI_IMG_HEIGHT,CFG.ORI_IMG_WIDTH])

            # convert binary image to json string
            img_name = name[idx]
            json_line = postprocess.postprocess_tensor(binary_img,img_name,lane_exist)

            json_str = json.dumps(json_line)
            file.write(json_str+'\n')

print('predict finished')

predict_json_path = '../evaluation/{}'.format(save_name)

res = LaneEval.bench_one_submit(predict_json_path,gt_json_path)
res = json.loads(res)
for r in res:
    print(r['name'], r['value'])







