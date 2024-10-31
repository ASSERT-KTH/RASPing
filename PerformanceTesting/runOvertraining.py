import sys
import os
import numpy as np
import jax.numpy as jnp

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.model import Model
from src.functions import *

#Argument parsing
import argparse

parser = argparse.ArgumentParser(description = "Training and save file arguments")

parser.add_argument("-baseModel", type=str, default="sort")
parser.add_argument("-maxLength", type=int, default=10)
parser.add_argument("-dataSize", type=int, default=5000)
parser.add_argument("-seed", type=int, default=666)

#parser.add_argument("-noiseType", type=str, default="bitFlip")
parser.add_argument("-noiseType", type=str, default="none") #!!!!! Fixed from the tested version
parser.add_argument("-noiseAmount", type=int, default=1)
parser.add_argument("-noiseParam", type=float, default=0.1)

parser.add_argument("-randomWeights", type=bool, default=True)
parser.add_argument("-n_epochs", type=int, default=50000)
parser.add_argument("-batch_size", type=int, default=256)
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-valStep", type=int, default=50)

parser.add_argument("-saveDirectory", type=str, default="savedData/overTraining/")
parser.add_argument("-lossFileName", type=str, default="temp_loss")
parser.add_argument("-valFileName", type=str, default="temp_val")

args = parser.parse_args()

print("Test run with arguments")
print("baseModel:", args.baseModel)
print("maxLength:", args.maxLength)
print("dataSize:", args.dataSize)
print("seed:", args.seed)

print("noiseType:", args.noiseType)
print("noiseAmount:", args.noiseAmount)
print("noiseParam:", args.noiseParam)

print("randomWeights:", args.randomWeights)
print("n_epochs:", args.n_epochs)
print("batch_size:", args.batch_size)
print("lr:", args.lr)
print("valStep:", args.valStep)

print("saveDirectory:", args.saveDirectory)
print("lossFileName:", args.lossFileName)
print("valFileName:", args.valFileName)

#Set up model and data
maxLength = args.maxLength
name = args.baseModel
N = args.dataSize

model = generateModel(name, maxLength)

model.setJaxPRNGKey(args.seed)
np.random.seed(args.seed)

#data = generateData(name, maxLength, N, args.randomWeights)    
data = generateData(name, maxLength, N, True) #!!!!! Fixed from the tested version
split = int(len(data) * 0.85)
data_train, data_val = data[:split], data[split:]
X, Y = encodeAndPadData(data, model.raspFunction, model.inputs, maxLength)

#Split data
split = int(X.shape[0] * 0.85)
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print("\n")
print("X shape: train", X_train.shape, ", val", X_val.shape)
print("Y shape: train", Y_train.shape, ", val", Y_val.shape)
print("Total unique samples", len(data))

noiseTypes = ["bitFlip", "gaussian", "flipFirst"]

#Train model
if args.randomWeights:
    model.setRandomWeights()
if args.noiseType in noiseTypes:
    model.addNoise(args.noiseType, args.noiseAmount, args.noiseParam)
metrics, validations = model.train(X_train, Y_train, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, plot=False, X_val=X_val, Y_val=Y_val, valStep=args.valStep)

#Save loss and validation accuracy
def saveArray(array, fileName="temp"):
    file = open(fileName, "wb")
    np.save(file, array)
    file.close()

saveDirectory = args.saveDirectory
saveArray([m["loss"] for m in metrics], saveDirectory + args.lossFileName)
saveArray(validations, saveDirectory + args.valFileName)