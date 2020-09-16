# SCNN-TF2
SCNN Tensorflow2实现版本

SCNN implemented by Tensorflow2

Paper Link:["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080), AAAI2018

Source Code:
"https://github.com/XingangPan/SCNN"

# Before Started

1. Clone the project

    ```
    git clone https://github.com/wind754203900/SCNN-TF2
    cd SCNN-TF2
    ```
2. Create a conda virtual environment and activate it

    ```
    conda create -n scnn_tf2 python=3.7 -y
    conda activate scnn_tf2
    ```
    
    Then install dependencies
    ```
    pip install -r requirements.txt
    ```
    

3. Data preparation

    Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract to`$TUSIMPLEROOT`. The directory arrangement of Tusimple should look         like:
    ```
    $TUSIMPLEROOT
    |-train_set
      |──clips
      |──label_data_0313.json
      |──label_data_0531.json
      |──label_data_0601.json
      |──readme.md
    |-test_set
      |──clips
      |──test_tasks_0627.json
      |──test_label.json
      |──readme.md
    ```
    Since the segmentation annotation is not provided for Tusimple, please generate seg segmentation from the json annotation. 
    ```
    cd data_provider
    python tusimple_processing.py   # modify variable of 'src_dir','dst_dir' and 'test_dir' in python file
    ```
    After running. You will get
    ```
    $TUSIMPLEROOT
    |-train_set
      |──...
      |──training
         |──train_instance.txt
         |──train_binary.txt
         |──validation_instance.txt
         |──validation_binary.txt
         |──gt_image
            |──....png
         |──gt_instance_image
            |──....png
         |──gt_binary_image
            |──....png
    |-test_set
      |──...
      |──test.txt
    ```
    
# Get Started
1. Train your model
    
    First, modify config file in `global_config/config.py`
    
    Change `TU_DATASETS_TRAIN` and `TU_DATASETS_VALID` to the path where your tusimple train and validation annotaion txt files store in
    ```
    # config file in global_config/config.py
    __C.TU_DATASETS_TRAIN = '{your_generated_tusimple_dataset_path}/training/train_instance.txt'
    __C.TU_DATASETS_VALID = '{your_generated_tusimple_dataset_path}/training/validation_instance.txt'
    ```
    
    OPTIONAL:
    
    You can change some training setting about `epoches,learning rate and so on` in config file
    
    Start training with: (You also can run the code in Pycharm)
    ```
    cd tools
    python train_deeplab_distribute.py
    ```
    Since I use `tf.distribute.MirroredStrategy()` in the code, the code will use one gpu or multi-gpus automatical automatically.
    
    After training, the weights file will be save in `weights/{model_name}.h5`
    
    You can also modify the model save path by editing `global_config/config.py`
    ```
    # config file in global_config/config.py
    __C.TRAIN.MODEL_SAVE_PATH = '{weigth_save_path_of_your_model}'.
    ```
    
2. Evaluation
    
    Before evaluation, you also should modify `global_config/config.py` to your tusimple test_dataset.
    
    ```
    # config file in global_config/config.py
    __C.TEST.DATAROOT = '{your_tusimple_test_dataset_path}/test_set'
    __C.TEST.ANNO_PATH = '{your_tusimple_test_dataset_path}/test_set/test.txt'
    ```
    
    If you want to visulize the result, please change the value `CFG.TEST.VISUALIZE ` to `True`
    ```
    # config file in global_config/config.py
    __C.TEST.VISUALIZE = True
    ```
#Other

The support for CULane Dataset will be implemented in the future.

