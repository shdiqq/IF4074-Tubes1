from function.generateImage import *

if __name__ == "__main__":
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  print("Data yang diperoleh dari generate image")
  print(dataInput[0].shape)
  print("===========")