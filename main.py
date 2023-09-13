import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

from function.generateImage import *
from CNN import CNN

if __name__ == "__main__":
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  # dataInput = np.array(
  #   [
  #     [
  #       [
  #         [1,11,2],
  #         [1,10,4],
  #         [6,12,8],
  #       ],
  #       [
  #         [7,1,2],
  #         [5,-1,2],
  #         [7,-4,2],
  #       ],
  #       [
  #         [-2,23,2],
  #         [2,20,4],
  #         [8,6,6],
  #       ]
  #     ]
  #   ]
  # )

  print("Data yang diperoleh dari generate image")
  print(dataInput[0].shape)
  print("===========")

  cnn = CNN(dataInput[0])

  cnn.addConvolutionalLayer(filterSize = 8, numFilter = 5, mode = 'max', padding = 2, stride = 8)

  print("Data yang diperoleh dari convolutional layer")
  print(cnn.output.shape)
  print("===========")

  cnn.addConvolutionalLayer(filterSize = 8, numFilter = 15, mode = 'max', padding = 2, stride = 8)

  print("Data yang diperoleh dari convolutional layer")
  print(cnn.output.shape)
  print("===========")

  cnn.addFlattenLayer()

  print("Data yang diperoleh dari flatten layer")
  print(cnn.output.shape)
  print("===========")

  cnn.addDenseLayer(outputSize = 16, activation = 'relu')

  print("Data yang diperoleh dari dense layer")
  print(cnn.output)
  print("===========")

  cnn.addDenseLayer(outputSize = 4, activation = 'relu')

  print("Data yang diperoleh dari dense layer")
  print(cnn.output)
  print("===========")