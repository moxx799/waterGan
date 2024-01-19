
import os
import pandas as pd
from Loader.dataset import CrackDatasetTF,get_train_augs,get_valid_augs,get_pred_augs
import tensorflow as tf

def construct_df(image_path,mask_path=None):
    if mask_path is None:
        image_files=os.listdir(image_path)
        df=pd.DataFrame(columns=['images'])
        df=pd.concat([pd.DataFrame([(image_path+image_files[i])], columns=['images']) for i in range(len(image_files))],
                ignore_index=True)
    else:
        image_files=os.listdir(image_path)
        gt_files=os.listdir(mask_path)
        df=pd.DataFrame(columns=['images','masks'])
        df=pd.concat([pd.DataFrame([(image_path+image_files[i],mask_path+gt_files[i])], columns=['images','masks']) for i in range(len(image_files))],
                ignore_index=True)
    
    return df

def config_datasets(train_images,gt_files,val_images,val_files=None,batch_size=1):
    
    if train_images is None:
        train_dataset = None
        val_df= construct_df(val_images,val_files)
        val_set = CrackDatasetTF(val_df)
        val_set = val_set.to_tf_dataset()
        val_dataset = val_set .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
    else:
        train_df= construct_df(train_images,gt_files)
        train_set = CrackDatasetTF(train_df)
        train_set = train_set.to_tf_dataset()
        train_dataset = train_set .batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
        val_df= construct_df(val_images,val_files)
        val_set = CrackDatasetTF(val_df)
        val_set = val_set.to_tf_dataset()
        val_dataset = val_set .batch(batch_size,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset,val_dataset


@tf.function
def compute_loss(logits, labels ,type='RMSE'):
    if type =='MSE':
        return tf.reduce_mean(tf.square(logits - labels))
    else:
        return tf.math.sqrt(tf.reduce_mean(tf.square(logits - labels)))

@tf.function
def compute_accuracy(logits, labels):
    # Modify this function according to your specific accuracy calculation needs
    # Example: Calculate the Euclidean distance and check if it's below a certain threshold
    threshold = 0.1
    distance = tf.abs((logits - labels))
    return tf.reduce_mean(tf.cast(tf.less(distance, threshold), tf.float32))

@tf.function
def compute_psnr(logits, labels):
        
    max_pixel = 2.0
    mse = tf.reduce_mean(tf.square(logits - labels))
    psnr = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)

    return psnr