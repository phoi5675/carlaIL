# carlaIL

autonomous driving model with multiple NNs using 1 front-facing camera and radar implemented on CARLA Simulator(0.9.11).

NNs is based on [End-to-end Driving via Conditional Imitation Learning
](https://arxiv.org/abs/1710.02410) and [YOLOv3 implemented by Tensorflow 1.X](https://github.com/YunYang1994/tensorflow-yolov3)

## sensors used 
- 1 front-facing rgb camera
- 1 front-facing radar

## what this driving model can do
- basic maneuvers; follow lanes, turn left / right or go straight at intersection, changing lanes using high-level command in CARLA
- detect red traffic lights and stop at stop at stop line.
- emergency braking(only for obstacles ahead)

because of the data collecting method and the limitation of based model, it can drive, **but not well**.

please see our paper for benchmark results(link in bottom).

# descriptions

collector : collect data from carla based on [carla imitation learning](https://github.com/carla-simulator/imitation-learning)

data order is same as link above, but the values may differ or not be collected.

imit_agent : running trained model in 0.9.X. to run our model

~~maual_data_collector.py, game.py is based on [manual_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py) and~~

~~automatic_data_controller.py is based on [automatic_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/automatic_control.py)~~

now merged : data_collector.py, please see args to collect data manually or automatically.

# how to collect data
- create "output" folder in collector/
- run data_collector.py
- 'R' key : toggle recording
- 'C' key : change weather

collected data will be saved in collecter/output/, named "data_#####.h5

for further information about Keyboard control or joystick(xbox one pad) control, please see collector/game.py

# how to train model
download [Imitation learning](https://github.com/merantix/imitation-learning)

overwrite repo with files in imitation-learning-train/imitation/ in this repo

to run the trainer, see readme file in link above and in imitation-learning-train/ in this repo.

# how to run model
see [benchmark_runner repo](https://github.com/phoi5675/benchmark-for-carla-imitation-learning)

this is the final version of our model.

# Requirements
- Carla 0.9.X (this code is based on ~~0.9.6~~ now using 0.9.11 due to radar support)
- PIL
- pygame
- numpy
- h5py
- tesnorflow==1.15.X or lower
- CUDA 10.0(depends on tensorflow version)

# known bugs
- model using tensorflow(for rocm), AMD gpu and ROCm in Ubuntu 18.04 might not drive as expected. 
this might be because I've tested on PC using Opencore EFI. running fine on windows and tensorflow-cpu in same environment or using Nvidia GPU(RTX 2080 super)
- there might be unexpected bugs :(

# acknowledgements
- carla for [CARLA Simulator](https://carla.org/)
- project's base autonomous driving model for [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)
- traffic light detection model for [YOLOv3 implemented by Tensorflow 1.X](https://github.com/YunYang1994/tensorflow-yolov3)

# paper
this is a project for undergraduate thesis, but the paper hasn't released on my university's library.

instead, I've uploaded on google drive.

[paper(korean)](https://drive.google.com/file/d/1Po2KdzNZ0QiEM0sU_TtCc9wesyc2q1hN/view?usp=sharing)

[presentation video(korean)](https://drive.google.com/file/d/13PeE7181RUUNDKQD5I01NV9hP1l8SX2M/view?usp=sharing)
