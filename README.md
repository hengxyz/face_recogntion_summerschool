# Demo of  Face Recogntion Inference
This demo shows how to use a pretrained deep CNNs model based on the embeddings distances between images to employ face recogntion

## Dependencies
- The code is tested on Ubuntu 16.04 with tf1.6 and python2.7 in CPU mode

## Configure the vitural enviroment for python:
- sudo apt-get install python-pip python-dev build-essential
- sudo apt-get install virtualenv virtualenvwrappewor
- vi ~/.bashrc
- adding to the end of the bashrc file: export WORKON_HOME=$HOME/.virtualenvs
- source /usr/local/bin/virtualenvwrapper.sh
- activating the configuration: source ~/.bashrc
- test: type 'workon' in the shell to see the current virtual enviroment

## Creat your own vitural enviroment
- mkvitualenv -p python=2 testdl_py2_tf1.6
- workon testdl_py2_tf1.6 (entering your own virtual env)

## Install the tools in the virtual enviroment:
- install Tensorflow 1.6: pip install tensorflow=1.6
- install opencv 2.4.13: pip install OPENCV-PYTHON
- install the package of the python : pip install scipy Pillow matplotlib sklearn h5py numpy matplotlib sklearn
- install tk package: sudo apt-get install python-tk (this will lead to exit the virtualenv, you should return to your virtualenv by workon testdl_py2_tf1.6 )
- check your enviroments:
1) Type python for launch python
2) >>>>> import tensorflow as tf
3) >>>>> tf.__version__ (it should show the version of the tf as tf 1.6)
4) >>>>> import cv2
5) >>>>> cv2.__version__ (it should show the version of the opencv)
6) >>>>> import numpy


## Download the Pretrained model
1) The pretrained model for face recogntion is here [](https://drive.google.com/drive/folders/1e5Ta__PVFaLEMYEFCXZrLUqOqXv2iNdT?usp=sharing)
2) put the download model under the folder ./face_recogntion_summerschool/models

### Examples for command line:
### Predict
1. CPU mode: python test_verifcation.py --model_dir ../models/20190629-011223/ --img_ref ../data/images/


