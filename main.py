from function.generateImage import *

import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'class')
sys.path.append(mymodule_dir)

# from layer.ConvolutionalLayer import ConvolutionalLayer
from stage.ConvolutionalStage import ConvolutionalStage
from stage.DetectorStage import DetectorStage

def main() :
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  print("Data yang diperoleh dari generate image")
  print(dataInput[0])
  print("===========")

  cnn = ConvolutionalStage(filterSize = (3, 3), numFilter = 3, numDepth = dataInput.shape[3], padding = 1, stride = 1)
  output = cnn.forward(dataInput[0])

  print("Data yang diperoleh setelah convolutional stage")
  print(output)
  print("===========")

  detectorStage = DetectorStage()
  outputDetector = detectorStage.forward(output)

  print("Setelah detector stage")
  print(outputDetector)
  print("===========")


main()