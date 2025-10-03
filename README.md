# waterGan

This repo provides a tensorflow trainer, tensor flow predictor and tensorflow Lite translation.

A example of the usage is in tf.ipynb
Loader is a dataloader that use the image path as the input, users can self-define the get_train_augs to do the augmentaions.
Models is the developed deep learning models


To train the model, use the `python train_gpus.py ` it will use the wandb as the train log record method, which means you need to know a little about wandb and have an account before using it.

To predict with the pretrained model , you need to change the path in the function to make it work.

The convert_model.ipynb provide the method to transfer a pretrained model to TFLite module.

# Raspberry-Pi models

Raspberry Pi system is installed in the micro-SD card, and you need to flash the system(image) with the latest version.
The official site provides a simple tutorial on how to install the image on it
`https://www.raspberrypi.com/software/`

After the image have been set up, we need to do several preparations for running deep learning model.

# 0. Check the camera and the system
After declare the system default python, and install the 3.11 version, you can check if the camera works.
In `~/cookbook` use `python capture.py` to get a sample piciture.

# 1. install the packages include tensorflow and pytorch

`$ sudo pip install torch torchvision torchaudio`
`$ sudo pip install opencv-python`
`$ sudo pip install numpy --upgrade`
reboot,then

`sudo pip install tensor flow`

# 2. clone the needed repositories.
Here are all the pre-trained models in this Pi

## 2.1 Deblur-Gan
https://github.com/VITA-Group/DeblurGANv2.git


## 2.2 Restormer
https://github.com/swz30/Restormer
Files are in the `~/Restormer` the deblur module is in the `Motion_deblurring/quanti.py`

## 2.3 tf models
https://github.com/sanghyun-son/ntire-2020-deblur-mobile
Files are in the `~/tfmodels`

We inplement uses this files which located in `~/color_enh/ /`
functions for real-time deblur & color enhancement
Remember to match the preprocessing steps for two models, some are normalize to [0,1] and some are [-1,1].

pred_h.py : deblur first and then color enhance
pred_h_v2.py : color enhance first and then deblur

functinons for predicting images to see the performance:

pred_images.py
pred_images_1co.py
pred_images_1co2de.py

Quantilize the model
quanti.py








