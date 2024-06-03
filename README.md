# waterGan

This repo provides a tensorflow trainer, tensor flow predictor and tensorflow Lite translation.

A example of the usage is in tf.ipynb
Loader is a dataloader that use the image path as the input, users can self-define the get_train_augs to do the augmentaions.
Models is the developed deep learning models


To train the model, use the `python train_gpus.py ` it will use the wandb as the train log record method, which means you need to know a little about wandb and have an account before using it.

To predict with the pretrained model , you need to change the path in the function to make it work.

The convert_model.ipynb provide the method to transfer a pretrained model to TFLite module.
