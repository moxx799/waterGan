from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization,  ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, Conv2D,GlobalAveragePooling2D
import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, ):
        super(Discriminator, self).__init__()        

        # Define layers in the constructor
        self.conv1 = Conv2D(16, kernel_size=3, strides=2, padding="same")
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.dropout1 = Dropout(0.25)
        self.conv2 = Conv2D(32, kernel_size=3, strides=2, padding="same")
        self.zero_padding = ZeroPadding2D(padding=((0,1),(0,1)))
        self.batch_norm = BatchNormalization(momentum=0.8)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)
        self.dropout2 = Dropout(0.25)
        self.conv3 = Conv2D(64, kernel_size=3, strides=2, padding="same")
        self.batch_norm2 = BatchNormalization(momentum=0.8)
        self.leaky_relu3 = LeakyReLU(alpha=0.2)
        self.dropout3 = Dropout(0.25)
        self.conv4 = Conv2D(128, kernel_size=3, strides=1, padding="same")
        self.batch_norm3 = BatchNormalization(momentum=0.8)
        self.leaky_relu4 = LeakyReLU(alpha=0.2)
        self.dropout4 = Dropout(0.25)
        self.max_pooling = MaxPooling2D()
        self.global_avg_pooling = GlobalAveragePooling2D((3,3))
        self.dense = Dense(1,activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1(x)
        x = self.max_pooling(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.max_pooling(x)
        x = self.zero_padding(x)
        x = self.batch_norm(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.max_pooling(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = self.max_pooling(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu4(x)
        x = self.dropout4(x)
        x = self.global_avg_pooling(x)
        x = self.dense(x)
        return x   