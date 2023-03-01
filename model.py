from keras.layers import merge, Conv3D, Input
from keras.layers import Input, concatenate 
from keras.models import Model
from tf.keras.layers import Conv3D,UpSampling3D, BatchNormalization, Dropout
# For skip connection we can use the below code
# merge3 = concatenate([conv1,up3], axis = 3)



def light_green_block(inp):
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Input_x1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha = 0.3)(x)
    
    return x
  
  
  def dark_green_block(x, filter_number):
    x = Conv3D(
        filters=filter_number,
        kernel_size=(3, 3, 3),
        strides=2,
        padding='same',
        data_format='channels_first',
        name='Input_x1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha = 0.3)(x)
    
    return x
  
  
  def blue_block():
    x = Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        strides=1,
        data_format='channels_first',
        name='Decoder_conv3d layer')(x4)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LeakyReLU(alpha = 0.3)(x)
    x = UpSampling3D(
        size=2,
        data_format='channels_first',
        name='Decoder_UpSample_256')(x)
    
    return x
  
  
  
 
