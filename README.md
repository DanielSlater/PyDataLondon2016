# Building a Pong playing AI

This repository contains the resources needed for the tutorial, Building a Pong playing AI in just 1 hour(Plus 4 days training time)

## Requirements

#### [Python](https://www.python.org/downloads/)
    Either 2 or 3 is fine.
#### [PyGame](http://www.pygame.org/download.shtml)
    Download which ever version matches the version of Python you plan on using.
#### [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup)
    Again match the version.
#### [Matplotlib](http://matplotlib.org/users/installing.html)
    Match version

## Other requirments options

Tensorflow requires an NVidea GPU and only runs on Linux/Mac so if you don't have these Theano is an option. The examples are all in Tensorflow, but that translates very easily to Theano and we have an example Q-learning Theano implementation that can be extended to work with Pong.

#### [Python](https://www.python.org/downloads/)
    Either 2 or 3 is fine.
#### [PyGame](http://www.pygame.org/download.shtml)
    Download which ever version matches the version of Python you plan on using.
#### [Theano](http://deeplearning.net/software/theano/install.html)
    Download anaconda
    
    On windows cmd:
    
    >> conda install mingw libpython numpy
    
    Checkout Theano
    
    git clone https://github.com/Theano/Theano.git
    
    >> cd Theano
    >> python setup.py develop
    
    Set your project interpreter to be using anaconda python


## Resources

#### [PyGame Player](https://github.com/DanielSlater/PyGamePlayer/blob/master/pygame_player.py)
    Used for running reinforcement learning agents against PyGame
#### [PyGame Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/pong.py)
    PyGame implementation of pong
#### [PyGame Half Pong](https://github.com/DanielSlater/PyGamePlayer/tree/master/games)
    Even pong can be hard if your just a machine. 
    Half pong is a simplified version of pong, if you can believe it.
    The score and other bits of noise are removed from the game. 
    There is only 1 bar and it is only 80x80 pixels which speeds up training and removes the need to downsize the screen 

## Examples

#### [Random Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/1_random_half_pong_player.py)
#### [Random With Base Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/2_random_with_base_half_pong_player.py)
#### [MLP Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/3_mlp_half_pong_player.py)
#### [Tensor flow Q learning](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/4_tensorflow_q_learning.py)
#### [Theano flow Q learning](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/4_theano_q_learning.py)
#### [MLP Q learning Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/5_mlp_q_learning_half_pong_player.py)
#### [Convolutional network Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/6_conv_net_half_pong_player.py)
