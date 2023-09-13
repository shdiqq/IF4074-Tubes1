import numpy as np
import os
import sys

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, '..', '..', 'function')
sys.path.append(mymodule_dir)

from activation import relu, sigmoid

class DenseLayer():
  def __init__(self, inputSize, outputSize, activation):
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.activation = activation
    self.weight = np.random.randn(inputSize, outputSize)
    self.bias = np.zeros((outputSize))

  def forward(self, inputData):
    output = np.dot(inputData, self.weight) + self.bias
    if (self.activation.lower() == 'relu'):
      output = relu(output)
    elif (self.activation.lower() == 'sigmoid'):
      output = sigmoid(output)
    return output
  
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
  print(matrix[0].shape)
  print("=====")
  matrix = np.ravel(matrix[0])
  print(matrix.shape)
  denseLayer = DenseLayer(inputSize = len(matrix), outputSize = 16, activation = 'sigmoid')
  newMatrix = denseLayer.forward(matrix)
  print(newMatrix)