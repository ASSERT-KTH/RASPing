import jax
import numpy as np
import matplotlib.pyplot as plt

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')

from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

#Calculates some weight statistic from a weight counter
def calculateWeightStatistics(weightCounter: dict, doPrint = False):
    totalValues = 0
    for _, n in weightCounter.items():
        totalValues+=n
    maxValue = max(weightCounter)
    minValue = min(weightCounter)
    zeroPercentage = 100*weightCounter[0]/totalValues if 0 in weightCounter else 0
    numberOfUniqueValues = len(weightCounter)

    if doPrint:
        print("N: %d\t min/max: %.2f/%.2f\t nValues: %d\t percentageZero: %.2f" % 
          (totalValues, minValue, maxValue, numberOfUniqueValues, zeroPercentage))
    return {"totalValues":totalValues, "maxValue": maxValue, "minValue": minValue, "zeroPercentage": zeroPercentage, "numberOfUniqueValues": numberOfUniqueValues}

#A class which holds the rasp models as well as a few helper functions and some statistics
class Model:
    def __init__(self, raspFunction: rasp.SOp, inputs, seqLength: int, name: str):
        self.raspFunction = raspFunction
        self.inputs = inputs
        self.seqLength = seqLength
        self.model = compiling.compile_rasp_to_model(self.raspFunction, self.inputs, self.seqLength, compiler_bos="BOS")
        self.name = name

        #Copy the inital weights in order to reset if required
        self.initialWeights = {}
        for name1, layer in self.model.params.items():
            self.initialWeights[name1] = {}
            for name2, weight in layer.items():
                self.initialWeights[name1][name2] = weight

        self.weightStatistics = {}
        self.updateWeightStatistics()

    #Reset weight to initial values
    def resetWeights(self):
        for name1, layer in self.model.params.items():
            for name2, _ in layer.items():
                self.model.params[name1][name2] = self.initialWeights[name1][name2]

    #Calculate and store new statistics for the weight distribution
    def updateWeightStatistics(self):
        self.weightStatistics = {}
        
        totalCounter = {}
        for name1, layer in self.model.params.items():
            self.weightStatistics[name1] = {}
            #print(name1, type(layer))
            for name2, weight in layer.items():
                weightCounter = {}
                #print("\t", name2, type(weight))

                #Find unique weights and count instances for the weights
                for t in weight.flatten():
                    t = float(t)
                    if t in weightCounter:
                        weightCounter[t]+=1
                    else:
                        weightCounter[t]=1

                #print("\t",end="  ")
                self.weightStatistics[name1][name2] = calculateWeightStatistics(weightCounter)

                #Appends the weight counts to the total counts
                for number, count in weightCounter.items():
                    if number in totalCounter:
                        totalCounter[number]+=count
                    else:
                        totalCounter[number]=count

        #print("\nTotal statistics")
        self.weightStatistics["total"] = calculateWeightStatistics(totalCounter)

    #Print the statistics for the weight distribution
    def printWeightStatistics(self, includeB=False):
        print(self.model.model_config)
        print("\nLayer analysis:")

        for name1, _ in self.weightStatistics.items():
            print(name1)
            if name1=="total":
                weightStats=self.weightStatistics[name1]
                print("\t  N: %d\t min/max: %.2f/%.2f\t nValues: %d\t percentageZero: %.2f" % 
                    (weightStats["totalValues"], weightStats["minValue"], weightStats["maxValue"], weightStats["numberOfUniqueValues"], weightStats["zeroPercentage"]))
                continue
            
            for name2, weightStats in self.weightStatistics[name1].items():
                if name2=="b" and includeB!=True:
                    continue
                print("\t", name2)
                print("\t  N: %d\t min/max: %.2f/%.2f\t nValues: %d\t percentageZero: %.2f" % 
                    (weightStats["totalValues"], weightStats["minValue"], weightStats["maxValue"], weightStats["numberOfUniqueValues"], weightStats["zeroPercentage"]))
    
    #Returns the boolean result for each case in the data set
    def evaluateModel(self, data, customName = None):
        if customName:
            print("Evaluating model:",customName)
        else:
            print("Evaluating model:",self.name)
        N=len(data)
        booleanAccuracy = np.zeros(N)
        
        for i in range(N):
            inputSeq, trueOutputSeq = data[i]
            outputSeq = self.apply(inputSeq)

            seqLength = len(trueOutputSeq)
            sameToken = np.zeros(seqLength)
            for ii in range(seqLength):
                sameToken[ii] = (outputSeq[ii]==trueOutputSeq[ii])
            
            booleanAccuracy[i] = (np.sum(sameToken) == seqLength)

            #TODO Add loading bar to keep track of progress

        return booleanAccuracy
    
    #Apply model to a sample
    def apply(self, input):
        return self.model.apply(input).decoded
    
    #Add noise to the model weights according too noiseType, amount and param
    def addNoise(self, noiseType = "bitFlip", amount=1, param = 0.1, includeEncoding = False):
        noiseTypes = ["bitFlip", "gaussian", "flipFirst", "temp"]
        if noiseType not in noiseTypes:
            print("Error: noiseType needs to be one of", noiseTypes)
            return
        
        match noiseType:
            #Flip binary bits 
            #If amount is a integer it flips that many random bits, if it is float it flips that fraction of bits
            case "bitFlip":
                #find binary weights in the model
                #Ensure that the weights are correctly changed before commiting to design. If assignment doesn't work I'll save the keys to access

                #Saves the keys to access all the layers with binary weights as well as the weight statistics for that layer
                binaryWeights = [] 
                totalCount = 0
                for name1, _ in self.weightStatistics.items():
                    if name1=="total":
                        continue
                    for name2, weightStats in self.weightStatistics[name1].items():
                        if weightStats["numberOfUniqueValues"]==2:
                            if includeEncoding == False and name2=="embeddings":
                                continue
                            binaryWeights.append((name1, name2, weightStats))
                            totalCount+=weightStats["totalValues"]

                if type(amount)==int:
                    #Randomly selects "amount" bits to flip
                    #The probability is equal for all applicable parameters
                    for i in range(amount):
                        #Parameter used to figure out the index where flip happens
                        index = np.random.randint(totalCount)
                        for name1, name2, stats in binaryWeights:
                            layerShape = self.model.params[name1][name2].shape
                            layerCount = layerShape[0]*layerShape[1]

                            #Check if this layer is the layer where the flip happens
                            if index>=layerCount:   #Not the correct layer
                                index-=layerCount
                                continue
                            
                            #Flip happens on this layer
                            index = (index//layerShape[1], index%layerShape[1])
                            self.model.params[name1][name2] = self.model.params[name1][name2].at[index[0],index[1]].set(
                                stats["maxValue"] - float(self.model.params[name1][name2][index[0],index[1]])
                            )

                            #print("Flip at:",name1,name2,index)

                            break

                elif type(amount)==float:
                    print("Percentage bitflip not yet implemented")
                    return
                else:
                    print("Error: amount needs to be int or float")

            case "gaussian":

                if type(amount)==int:
                    print("Counted gaussian not yet implemented")
                    return
                
                #Adds gaussian noise with standard deviation "param" to "amount" fraction of the weights
                elif type(amount)==float:
                    for name1, _ in self.weightStatistics.items():
                        if name1=="total":
                            continue
                        for name2, weightStats in self.weightStatistics[name1].items():
                            if name2=="b":
                                continue
                            if includeEncoding == False and name2=="embeddings":
                                continue

                            layerShape = self.model.params[name1][name2].shape
                            self.model.params[name1][name2] = self.model.params[name1][name2] + \
                                np.where(np.random.rand(layerShape[0], layerShape[1])<amount, np.random.normal(0, param, layerShape), 0)
                            
                    return
                else:
                    print("Error: amount needs to be int or float")
            
            #Basic test to simply flip first attention key weight
            case "flipFirst":
                self.model.params["transformer/layer_0/attn/key"]["w"] = self.model.params["transformer/layer_0/attn/key"]["w"].at[0,0].set(1)
                #weights = model.model.params["transformer/layer_0/attn/key"]["w"]   #This assignment does not work. Guessing it copies due to strict immutability
                #weights = weights.at[0,0].set(1)
                print(self.model.params["transformer/layer_0/attn/key"]["w"][0,0])
                print(self.model.params["transformer/layer_0/attn/key"]["w"])

            #Currently adds one to first key weights
            case "temp":
                shape = self.model.params["transformer/layer_0/attn/key"]["w"].shape
                print(shape)
                self.model.params["transformer/layer_0/attn/key"]["w"] = self.model.params["transformer/layer_0/attn/key"]["w"]+np.ones(shape) 
                print("Warning: Temporary noise mode used!")
                return


            case _:
                print("Error: noiseType not implemented")

        
        return
    
