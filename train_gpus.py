from Models.U_net_tf import RSUNet
from Models.Ntire import Baseline
import tensorflow as tf
import os
import wandb
from wandb.keras import WandbCallback
# from wandb.keras import WandbMetricsLogger
# from wandb.keras import WandbModelCheckpoint
from utils.utils import config_datasets,compute_accuracy,compute_loss,compute_psnr
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--debug_mode", dest="debug_mode", type=bool,default=False,
                        help="debug mode or not")
    parser.add_argument("-n", "--wandb_name", dest="wandb_name", type=str,default=None,
                        help="Give a name to save checkpoints")
    parser.add_argument("-t", "--wandb_tag", dest="wandb_tag", type=str,default=None,
                        help="Give a tag to your running")
    

    args = parser.parse_args()

    return args


class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        logs['learning_rate'] = self.model.optimizer.lr(self.model.optimizer.iterations).numpy()
        wandb.log(logs)

if __name__=="__main__":
    
    args=read_options() 
    initial_learning_rate = 0.001
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True)    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)    
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)

    # Define the strategy based on the number of GPUs
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        print(f'Using MirroredStrategy with {num_gpus} GPUs.')
    else:
        strategy = tf.distribute.OneDeviceStrategy("GPU:0")
        print('Using OneDeviceStrategy with a single GPU.') 
    
    with strategy.scope():  
        model = RSUNet(layers=5, filters=[16,32, 64, 128, 256, 256], output_channels=3)
        #model= Baseline(512,512)
        model.compile(optimizer=optimizer, 
                    loss=compute_loss,
                    metrics=[compute_accuracy,compute_psnr])
    # end of the strategy            
    
    if args.debug_mode is True:
        # debug mode use the val as train set, to speed up the process.
        train_images='./Data/val/blur/blur/'
        gt_files='./Data/val/sharp/sharp/'
    else:
        train_images='./Data/train/blur/blur/'
        gt_files='./Data/train/sharp/sharp/'
    val_images='./Data/val/blur/blur/'
    val_files='./Data/val/sharp/sharp/'
    
    run = wandb.init(project="waterGan", entity='moxx799',name= args.wandb_name,notes=args.wandb_tag)
    config = wandb.config
    config.batch_size = 16
    config.epochs = 40  
    working_dir=os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(working_dir, "Checkpoints", args.wandb_name), exist_ok=True)
    checkpoint_path = os.path.join(working_dir, "Checkpoints", args.wandb_name,)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        save_weights_only=False, 
        save_best_only=True
    )
    train_dataset,val_dataset = config_datasets(train_images,gt_files,val_images,val_files=val_files,batch_size=config.batch_size)
    wandb_callback=WandbCallback(save_model=True)
    learning_rate_logger = LearningRateLogger()
    
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback,learning_rate_logger,wandb_callback] # if you have callbacks checkpoint_callback, WandbCallback(),learning_rate_logger
    )
    run.finish()