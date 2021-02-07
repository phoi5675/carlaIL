# carlaIL


**please be aware that there are some bugs**


collector : collect data from carla based on [carla imitation learning](https://github.com/carla-simulator/imitation-learning)

data order is same as above page, but the values may differ or not be collected.

imit_agent : running trained model in 0.9.X (in devlopment)

~~maual_data_collector.py, game.py is based on [manual_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py) and~~

~~automatic_data_controller.py is based on [automatic_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/automatic_control.py)~~

now merged : data_collector.py, please see args to collect data manually or automatically.

you have to create folder "output" in collector/

collected data will be saved in collecter/output/, named "data_#####.h5

when you press 'R' key, which is toggle button for recording, it will begin collecting data.

for further information about Keyboard control or joystick(xbox one pad) control, please see collector/game.py


# Requirements
- Carla 0.9.X (this code is based on 0.9.6)
- PIL
- pygame
- numpy
- h5py
