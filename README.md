# Building a Pong playing AI

This repository contains the resources needed for the tutorial, Building a Pong playing AI in just 1 hour(Plus 4 days training time)

## Requirements

#### [Linux](https://www.linuxmint.com/download.php)
    All the examples are in tensorflow which currently does not have a windows version. 
    If you don't fancy installing linux right now...
    It is possible to build everything here in [Theano](http://deeplearning.net/software/theano/) while you follow the talk.
    If anyway does any of this successfully in theano please submit it :)
#### [Python](https://www.python.org/downloads/)
    Ether 2 or 3 is fine.
#### [PyGame](http://www.pygame.org/download.shtml)
    Download which ever version matches the version of Python you plan on using.
#### [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup)
    Again match the version.

## Resources

#### [PyGame Player](https://github.com/DanielSlater/PyGamePlayer/blob/master/pygame_player.py)
    Used for running reinforcement learning agents against PyGame
#### [PyGame Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/pong.py)
    PyGame implementation of pong
#### [PyGame Half Pong](https://github.com/DanielSlater/PyGamePlayer/tree/master/games)
    Even pong can be hard if your just a machine. 
    Half pong is a simplified version of pong, if you can believe that is possible.
    The score and other bits of noise are removed from the game. 
    There is only 1 bar and it is only 80x80 pixels which speeds up training and removes the need to downsize the screen 

## Examples

#### Random Half Pong player
#### MLP Half Pong player
#### Tensor flow Q learning
#### MLP Q learning Half Pong player
#### Convolutional network Half Pong player
#### Experience replay + Explore Half Pong player
#### Full pong player