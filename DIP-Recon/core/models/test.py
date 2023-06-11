import torch
from .skip import skip
from .light_cnn import ConvDecoder

def getModel1():
  num_channels = 256

  model = skip([640,372],num_channels, 30, 
  num_channels_down = [num_channels] * 8,
  num_channels_up =   [num_channels] * 8,
  num_channels_skip =    [num_channels*0] * 6 + [4,4],  
  filter_size_up = 3, filter_size_down = 5, 
  upsample_mode='nearest', filter_skip_size=1,
  need_sigmoid=False, need_bias=True, pad='zero', act_fun='ReLU')  
  return model

def getModel2():
  return ConvDecoder(7, 256, 30, [640,372],
              [8,4], 'ReLU', 'Nearest', 'bn', False,False)  
