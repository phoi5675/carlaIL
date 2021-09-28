# description
this is a repo for training our model based on [Imitation learning](https://github.com/merantix/imitation-learning).

training model based on collected data and high-level command from CARLA 0.9.X

# how to train
- download original repo and overwrite files from this repo
- to do data preprocessing and start training : see original repo's readme
- to train lane changing model, which has 4 branches based on high-level commands; 3: STRAIGHT, 4: LANEFOLLOW, 5: CHANGELEFT, 6: CHANGERIGHT, 
change [this code](https://github.com/phoi5675/carlaIL/blob/main/imitation-learning-train/imitation/preprocessor.py#L147) to
```
if intention is None or (hlc == 1.0 or hlc == 2.0) or (hlc == 4.0 and rand <= 0.6):
```
