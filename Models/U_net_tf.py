import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation

class ConvBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2D(filters, 7, padding='same',kernel_initializer='he_normal')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(filters, 3, padding='same',kernel_initializer='he_normal')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class EncoderBlock(tf.keras.Model):
    def __init__(self, filters):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(filters)
        self.pool = MaxPooling2D(2)

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.pool(x)
        
        return x, p


class DecoderBlock(tf.keras.Model):
    def __init__(self, filters):
        super(DecoderBlock, self).__init__()
        self.up = Conv2DTranspose(filters, 2, strides=2, padding='same',kernel_initializer='he_normal')
        self.conv_block = ConvBlock(filters)

    def call(self, inputs, skip):
        x = self.up(inputs)
        x = concatenate([x, skip])
        x = self.conv_block(x)
        return x

class UNet(tf.keras.Model):
    def __init__(self, layers, filters,output_channels):
        super(UNet, self).__init__()
        self.encoder_blocks = [EncoderBlock(filters[i]) for i in range(layers)]
        self.center = ConvBlock(filters[layers])
        self.decoder_blocks = [DecoderBlock(filters[i]) for i in range(layers-1, -1, -1)]
        self.final_conv = Conv2D(output_channels, 1, activation='tanh',kernel_initializer='he_normal')

    def call(self, inputs):
        skips = []
        x = inputs
        for encoder in self.encoder_blocks:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.center(x)

        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)
    
class RSUNet(tf.keras.Model):
    def __init__(self, layers, filters, output_channels):
        super(RSUNet, self).__init__()
        self.encoder_blocks = [EncoderBlock(filters[i]) for i in range(layers)]
        self.center = ConvBlock(filters[layers])
        self.decoder_blocks = [DecoderBlock(filters[i]) for i in range(layers-1, -1, -1)]
        self.final_conv = Conv2D(output_channels, 1, activation='tanh',kernel_initializer='he_normal')

    def call(self, inputs):
        skips = []
        x = inputs
        for encoder in self.encoder_blocks:
            skip, x = encoder(x)
            skips.append(skip)

        x = self.center(x)

        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder(x, skip)

        # Add the input image to the output
        output = self.final_conv(x)
        return tf.clip_by_value(inputs + output, 0,1)  # Adding the input to the final output

