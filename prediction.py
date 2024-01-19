import tensorflow as tf
# from Models.U_net_tf import 
from Models.Ntire import Baseline
from utils.utils import construct_df, config_datasets, compute_accuracy, compute_loss
import os
import numpy as np
import albumentations as A

working_dir=os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
val_images_path = './Data/pred/blur/'
val_labels_path = './Data/pred/sharp/'
predicted_path = './Data/pred/predicted/'
checkpoint_name='Testbaseline'

checkpoint_path = os.path.join(working_dir, "Checkpoints",checkpoint_name) # "./Checkpoints/best_model"
prediction_batch_size=32

# Create the predicted path if it does not exist
if not os.path.exists(predicted_path):
    os.makedirs(predicted_path)

# Load the model
#model = RSUNet(layers=4, filters=[32, 64, 128, 256, 256], output_channels=3)
model= Baseline(512,512)
model.compile(optimizer='adam', loss=compute_loss, metrics=[compute_accuracy])

# Load weights

model.load_weights(checkpoint_path)

# Prepare the validation dataset
# Assuming config_datasets function properly configures and returns the dataset
_, val_dataset = config_datasets(None, None, val_images_path,val_labels_path, batch_size=prediction_batch_size)# can add sharp images with val_labels_path

# Predict and evaluate
total_accuracy = 0
num_batches = 0
for batch, (images, labels) in enumerate(val_dataset):
    predictions = model.predict(images)
    
    # Compute accuracy for the batch
    batch_accuracy = compute_accuracy(labels, predictions)
    total_accuracy += batch_accuracy
    num_batches += 1

    # Save predictions
    for i, prediction in enumerate(predictions):
        # Convert prediction to appropriate format if necessary
        prediction = ((prediction+1)*127.5).astype(np.uint8)
        prediction = A.resize(prediction,512,512)
        save_path = os.path.join(predicted_path, f"prediction_{batch}_{i}.png")
        tf.keras.preprocessing.image.save_img(save_path, prediction)

# Compute overall accuracy
overall_accuracy = total_accuracy / num_batches
print(f"Validation Accuracy: {overall_accuracy}")

# The predicted images are saved in the 'predicted_path' directory

