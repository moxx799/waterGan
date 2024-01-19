import tensorflow as tf
import numpy as np
import albumentations as A

def get_train_augs():
    # Define augmentations
    return A.Compose([
        A.RandomCrop(height=256, width=256),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.VerticalFlip(p=0.5)
    ], additional_targets={'mask': 'image'})

def get_valid_augs():
    # Define validation augmentations
    return A.Compose([
        A.RandomCrop(height=256, width=256)
    ], additional_targets={'mask': 'image'})
    
def get_pred_augs():
    # Define validation augmentations
    return A.Compose([
        
    ], additional_targets={'mask': 'image'})

class CrackDatasetTF():
    def __init__(self, df, augmentations=False):
        self.df = df
        self.augmentations = augmentations

    def _load_and_preprocess(self, image_path, mask_path):
        # Load the image and mask
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image * 2 - 1

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        mask = mask * 2 - 1

        if self.augmentations:
            # Apply augmentations
            augmented = self.augmentations(image=image.numpy(), mask=mask.numpy())
            image, mask = tf.convert_to_tensor(augmented['image']), tf.convert_to_tensor(augmented['mask'])

        return image, mask

    def __len__(self):
        return len(self.df)

    def to_tf_dataset(self):
        def gen():
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                image_path, mask_path = row.images, row.masks
                yield self._load_and_preprocess(image_path, mask_path)

        return tf.data.Dataset.from_generator(
            gen, 
            output_types=(tf.float32, tf.float32), 
            output_shapes=((None, None, 3), (None, None, 3))
        )
        
# def get_train_augs():
#     return A.Compose([
#         A.RandomCrop(height=256, width=256),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.VerticalFlip(p=0.5)
#     ])

# def get_valid_augs():
#     return A.Compose([A.RandomCrop(height=256, width=256),    
#     ])
# Usage
# crack_dataset = CrackDatasetTF(df, augmentations)
# tf_dataset = crack_dataset.to_tf_dataset()
# tf_dataset = tf_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
