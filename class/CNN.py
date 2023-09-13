import os
import sys
import numpy as np

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.FlattenLayer import FlattenLayer
from layer.DenseLayer import DenseLayer

class CNN():
  def __init__(self, input):
    self.input = input
    self.output = None
    self.inputSize = 0
    self.outputSize = 0

  def addConvolutionalLayer(self, filterSize, numFilter, mode, padding = 0, stride = 1):
    convolutionalLayer = ConvolutionalLayer(self.input.shape, filterSize, numFilter, mode, padding, stride)
    self.output = convolutionalLayer.forward(self.input)
    self.inputSize = (self.output).shape
    self.outputSize = (self.output).shape

  def addFlattenLayer(self):
    flattenLayer = FlattenLayer()
    self.output = flattenLayer.forward(self.output)
    self.inputSize = (self.output).shape
    self.outputSize = (self.output).shape

  def addDenseLayer(self, outputSize, activation):
    denseLayer = DenseLayer(self.inputSize[0], outputSize, activation)
    self.output = denseLayer.forward(self.output)
    self.inputSize = (self.output).shape
    self.outputSize = (self.output).shape

### TESTING ###
if __name__ == "__main__":
  matrix = np.array(
    [
      [
        [
          [1,11,2],
          [1,10,4],
          [6,12,8],
        ],
        [
          [7,1,2],
          [5,-1,2],
          [7,-4,2],
        ],
        [
          [-2,23,2],
          [2,20,4],
          [8,6,6],
        ]
      ]
    ]
  )

  cnn = CNN(matrix[0])
  cnn.addConvolutionalLayer(filterSize = 2, numFilter = 3, mode = 'max', padding = 1, stride = 1)
  print(cnn.outputSize)
  cnn.addConvolutionalLayer(filterSize = 2, numFilter = 6, mode = 'average', padding = 1, stride = 1)
  print(cnn.outputSize)
  cnn.addFlattenLayer()
  print(cnn.outputSize)
  cnn.addDenseLayer(outputSize = 16, activation = 'relu')
  cnn.addDenseLayer(outputSize = 4, activation = 'relu')
  cnn.addDenseLayer(outputSize = 8, activation = 'relu')
  print(cnn.output)