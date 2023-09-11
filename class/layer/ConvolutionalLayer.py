import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'stage')
sys.path.append(mymodule_dir)

from ConvolutionalStage import ConvolutionalStage

class ConvolutionalLayer():
  def __init__(self, filterSize, numFilter,  numDepth, padding = 0, stride = 1):
    self.convolution_stage = ConvolutionalStage(filterSize, numFilter, numDepth, padding, stride)

  def forward(self, inputData):
    feature_map = self.convolution_stage.forward(inputData)
    return feature_map