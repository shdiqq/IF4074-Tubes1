import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'stage')
sys.path.append(mymodule_dir)

from ConvolutionalStage import ConvolutionalStage
from DetectorStage import DetectorStage

class ConvolutionalLayer():
  def __init__(self, filterSize, numFilter,  numDepth, padding = 0, stride = 1):
    self.convolutionStage = ConvolutionalStage(filterSize, numFilter, numDepth, padding, stride)
    self.detectorStage = DetectorStage()

  def forward(self, inputData):
    featureMap = self.convolutionStage.forward(inputData)
    outputDetector = self.detectorStage.forward(featureMap)
    return outputDetector