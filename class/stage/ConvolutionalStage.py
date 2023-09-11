import numpy as np

class ConvolutionalStage():
  def __init__(self, filterSize, numFilter, numDepth, padding = 0, stride = 1):
    self.filterSize = filterSize
    self.numFilter = numFilter
    self.numDepth = numDepth
    self.padding = padding
    self.stride = stride
    self.bias = np.zeros((numFilter))
    self.kernel = np.random.randn(self.numFilter, self.filterSize[0], self.filterSize[1], self.numDepth)

  # def addZeroPadding(self, inputData):
  #   inputDataUpdate = np.pad(inputData, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
  #   return inputDataUpdate

  def getOutputSize(self, inputHeight, inputWidth):
    outputHeight = ( inputHeight - self.filterSize[0] + 2 * self.padding ) // self.stride + 1
    outputWidth = ( inputWidth - self.filterSize[1] + 2 * self.padding ) // self.stride + 1
    return outputHeight, outputWidth
        
  def forward(self, inputData):
    print("Proses forward pada Convolutional Stage")
    # print("Awal")
    # print(inputData.shape)
    # print("===========")

    # inputDataUpdate = self.addZeroPadding(inputData)
    # print("Setelah padding")
    # print(inputDataUpdate.shape)
    # print("===========")

    inputHeight, inputWidth, inputDepth = inputData.shape
    outputHeight, outputWidth = self.getOutputSize(inputHeight, inputWidth)
    featureMap = np.zeros((outputHeight, outputWidth, self.numFilter))

    for i in range(self.numFilter) :
      for row in range(0, inputHeight - self.filterSize[0] + 1, self.stride) :
        for col in range(0, inputWidth - self.filterSize[1] + 1, self.stride) :
          inputPatch = inputData[row : row + self.filterSize[0], col : col + self.filterSize[1], :]
          featureMap[row // self.stride, col // self.stride, i] = np.sum(inputPatch * self.kernel[i]) + self.bias[i]

    return featureMap