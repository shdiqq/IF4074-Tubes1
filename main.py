import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from function.generateImage import *
from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.DenseLayer import DenseLayer
from layer.FlattenLayer import FlattenLayer

if __name__ == "__main__":
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  print("Data yang diperoleh dari generate image")
  print(dataInput[1].shape)
  print("===========")

  convolutionalLayer = ConvolutionalLayer(inputSize = dataInput[0].shape, filterSize = 16, numFilter = 3, mode='max', padding = 1, stride = 8)
  newMatrix = convolutionalLayer.forward(dataInput[0])

  print("Data yang diperoleh dari convolutional layer")
  print(newMatrix.shape)
  print("===========")

  flattenLayer = FlattenLayer()
  newMatrix = flattenLayer.forward(newMatrix)

  print("Data yang diperoleh dari flatten layer")
  print(newMatrix.shape)
  print("===========")

  denseLayer = DenseLayer(newMatrix.shape[0], 125, 'relu')
  newMatrix = denseLayer.forward(newMatrix)

  print("Data yang diperoleh dari dense layer")
  print(newMatrix.shape)
  print("===========")