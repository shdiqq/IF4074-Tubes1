import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'stage')
sys.path.append(mymodule_dir)

from ConvolutionalStage import ConvolutionalStage
from DetectorStage import DetectorStage
from PoolingStage import PoolingStage

class ConvolutionalLayer():
  def __init__(self, filterSize, numFilter, numDepth, mode, padding = 0, stride = 1):
    self.convolutionStage = ConvolutionalStage(filterSize, numFilter, numDepth, padding, stride)
    self.detectorStage = DetectorStage()
    self.poolingStage = PoolingStage(filterSize, stride, mode)

  def forward(self, inputData):
    featureMap = self.convolutionStage.forward(inputData)
    outputDetector = self.detectorStage.forward(featureMap)
    outputPooling = self.poolingStage.forward(outputDetector)
    return outputPooling