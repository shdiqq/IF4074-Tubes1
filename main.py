from function.generateImage import *

def main() :
  objectClassDictionary = {
    0: 'bear',
    1: 'panda'
  }
  dataInput, dataClassLabel = generateImage()

  print("Data yang diperoleh dari generate image")
  print(dataInput[0])
  print("===========")

main()