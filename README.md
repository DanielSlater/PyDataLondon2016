# Building a Pong playing AI

This repository contains the resources needed for the tutorial, Building a Pong playing AI in just 1 hour(Plus 4 days training time)

### Installation Guide for OS X

Tested on a Macbook Pro (late 2013) with El Capitan, unsure if GPU-support works.

Requirements:
* [Homebrew](http://brew.sh/)
* [Miniconda](http://conda.pydata.org/miniconda.html)


Install some image libraries and a X framework for MacOS:

```sh
brew install sdl_image
brew install Caskroom/cask/xquartz
```

Clone the repo:

```
git clone git@github.com:DanielSlater/PyDataLondon2016.git
cd PyDataLondon2016/
```

Create a virtual environment for Python 2:

```
conda create --name pong-ai-27 python=2
source activate pong-ai-27
```

Install listed dependencies plus `opencv`:

```sh
conda install matplotlib numpy opencv
```

Install `tensorflow` and `pygame`:

```sh
conda install -c https://conda.anaconda.org/jjhelmus tensorflow
conda install -c https://conda.binstar.org/quasiben pygame
```


Initialize submodules:

```
git submodule init
git submodule update
```

Symlink `resources` and `common` in folder `examples`:

```
cd examples/
ln -s ../resources/ 
ln -s ../common/ 
```


Run an example:

```
python 1_random_half_pong_player.py
```

### Linux Nvidea GPU installation Guide

* [Python 2](https://www.python.org/downloads/)
* [PyGame](http://www.pygame.org/download.shtml)
* [TensorFlow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#download-and-setup)
* [Matplotlib](http://matplotlib.org/users/installing.html)

Tensorflow requires an NVidia GPU and only runs on Linux/Mac so if you don't have these Theano is an option (see below). The examples are all in Tensorflow, but that translates very easily to Theano and we have an example Q-learning Theano implementation that can be extended to work with Pong.

## Windows/non nvidia gpu

#### [Python](https://www.python.org/downloads/)
    Either 2 or 3 is fine.
#### [PyGame](http://www.pygame.org/download.shtml)
    Download which ever version matches the version of Python you plan on using.
#### [Matplotlib](http://matplotlib.org/users/installing.html)
    Match version

### [Theano Installation Guide for Windows](http://deeplearning.net/software/theano/install.html)

Download anaconda and install packages:

```
conda install mingw libpython numpy
```

Clone Theano repo:

```
git clone https://github.com/Theano/Theano.git
```

Install theano package:

```
cd Theano
python setup.py develop
```

###Docker environment alternative
#### Docker build
    Have a look at the Makefile, essentially this helps you setup an xquartz environment exposed to a docker container along with the required dependencies.
    'make all' should in theory launch you into an environment capable of running th examples straight away.

## Resources

#### [PyGame Player](https://github.com/DanielSlater/PyGamePlayer/blob/master/pygame_player.py)
Used for running reinforcement learning agents against PyGame

#### [PyGame Pong](https://github.com/DanielSlater/PyGamePlayer/blob/master/games/pong.py)
PyGame implementation of pong

#### [PyGame Half Pong](https://github.com/DanielSlater/PyGamePlayer/tree/master/games)
Even pong can be hard if you're just a machine. 
Half pong is a simplified version of pong, if you can believe it.
The score and other bits of noise are removed from the game. 
There is only 1 bar and it is only 80x80 pixels which speeds up training and removes the need to downsize the screen 


## Examples

* [Random Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/1_random_half_pong_player.py)
* [Random With Base Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/2_random_with_base_half_pong_player.py)
* [MLP Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/3_mlp_half_pong_player.py)
* [Tensor flow Q learning](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/4_tensorflow_q_learning.py)
* [Theano flow Q learning](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/4_theano_q_learning.py)
* [MLP Q learning Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/5_mlp_q_learning_half_pong_player.py)
* [Convolutional network Half Pong player](https://github.com/DanielSlater/PyDataLondon2016/blob/master/examples/6_conv_net_half_pong_player.py)
