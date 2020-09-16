# SCNN-TF2
SCNN Tensorflow2实现版本

SCNN implemented by Tensorflow2

Paper Link:["Spatial As Deep: Spatial CNN for Traffic Scene Understanding"](https://arxiv.org/abs/1712.06080), AAAI2018

Source Code:
"https://github.com/XingangPan/SCNN"

# Before Started

1. Data preparation

Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract to`$TUSIMPLEROOT`. The directory arrangement of Tusimple should look like:
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
