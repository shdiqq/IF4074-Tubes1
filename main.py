from function.generateImage import *

import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

# from layer.ConvolutionalLayer import ConvolutionalLayer
from layer.ConvolutionalLayer import ConvolutionalLayer

def main() :
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  print("Data yang diperoleh dari generate image")
  print(dataInput[0])
  print("===========")

  cnn = ConvolutionalLayer(filterSize = (3, 3), numFilter = 3, numDepth = dataInput.shape[3], stride = 1, padding = 1)
  output = cnn.forward(dataInput[0])

  print("Data yang diperoleh setelah convolutional stage")
  print(output)
  print("===========")

main()