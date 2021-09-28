# carlaIL


**please be aware that there are some bugs**

# descriptions

collector : collect data from carla based on [carla imitation learning](https://github.com/carla-simulator/imitation-learning)

data order is same as above page, but the values may differ or not be collected.

imit_agent : running trained model in 0.9.X. to run our model

~~maual_data_collector.py, game.py is based on [manual_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py) and~~

~~automatic_data_controller.py is based on [automatic_control.py](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/automatic_control.py)~~

now merged : data_collector.py, please see args to collect data manually or automatically.

collected data will be saved in collecter/output/, named "data_#####.h5

when you press 'R' key, which is toggle button for recording, it will begin collecting data.

to change weather, use 'C'

for further information about Keyboard control or joystick(xbox one pad) control, please see collector/game.py

# how to collect data
- create "output" folder in collector/
- run data_collector.py
- 'R' key : toggle recording
- 'C' key : change weather

# how to train model
see 

# how to run model
see [benchmark_runner repo](https://github.com/phoi5675/benchmark-for-carla-imitation-learning)

this is the final version of our model.

# Requirements
- Carla 0.9.X (this code is based on ~~0.9.6~~ now using 0.9.11 due to radar support)
- PIL
- pygame
- numpy
- h5py
