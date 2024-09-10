import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import optax
import pandas as pd

from typing import NamedTuple
import haiku as hk

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "float32")

from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

from .loss import *
from .trainer import Trainer

#Global forward functions which gets set to the appropriate function each time it needs to be called from a class instance
forward = None
forward_fun = None

class GridSearchParameters():
    def __init__(self):
        self.epochs = []


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
        self.raspFunction = raspFunction
        self.model = compiling.compile_rasp_to_model(self.raspFunction, self.inputs, self.seqLength, compiler_bos="BOS")
        self.name = name

        #Copy the inital weights in order to reset if required
        self.initialWeights = {}
        for name1, layer in self.model.params.items():
            self.initialWeights[name1] = {}
            for name2, weight in layer.items():
                self.initialWeights[name1][name2] = weight

        self.jaxPRNGKey = jax.random.key(666)

        self.weightStatistics = {}
        self.updateWeightStatistics()

        self.setForwardFun()

    def setForwardFun(self):
        global forward
        def forward(x):
            compiled_model = self.model.get_compiled_model()
            compiled_model.use_unembed_argmax = False
            compiled_model.pad_token = self.model.input_encoder.encoding_map["compiler_pad"]
            return compiled_model(x, use_dropout=False)

        global forward_fun
        forward_fun = hk.without_apply_rng(hk.transform(forward))

    def setJaxPRNGKey(self, newSeed):
        self.jaxPRNGKey = jax.random.key(newSeed)

    #Reset weight to initial values
    def resetWeights(self):
        for name1, layer in self.model.params.items():
            for name2, _ in layer.items():
                self.model.params[name1][name2] = self.initialWeights[name1][name2]

    def setRandomWeights(self, mean=0.0, std=1.0):     
        self.jaxPRNGKey, newPRNGKey = jax.random.split(self.jaxPRNGKey)
        PRNGSeq = hk.PRNGSequence(newPRNGKey)
        randomParams = jax.tree_util.tree_map(
            lambda p: jax.random.normal(next(PRNGSeq), p.shape) * std + mean, self.model.params
        )
        self.model.params = randomParams

    #Sets the model weights to 'params'
    def setWeights(self, params):
        self.model.params = params

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
    def evaluateModel(self, data, customName = None, doPrint = True, outputArray = True):
        self.setForwardFun()

        if doPrint:
            if customName:
                print("Evaluating model:",customName)
            else:
                print("Evaluating model:",self.name)

        N=len(data)
        if outputArray:
            booleanAccuracy = np.zeros(N)
        else:
            booleanAccuracy = 0
        
        for i in range(N):
            inputSeq, trueOutputSeq = data[i]
            outputSeq = self.apply(inputSeq)

            seqLength = len(trueOutputSeq)
            sameToken = np.zeros(seqLength)
            for ii in range(seqLength):
                sameToken[ii] = (outputSeq[ii]==trueOutputSeq[ii])
            
            if outputArray:
                booleanAccuracy[i] = (np.sum(sameToken) == seqLength)
            else:
                booleanAccuracy += (np.sum(sameToken) == seqLength)

            #TODO Add loading bar to keep track of progress

        if outputArray:
            return booleanAccuracy
        else:
            return booleanAccuracy / N

    def fastEvaluateEncoded(self, X, Y):
        padToken = self.model.input_encoder.encoding_map["compiler_pad"]
        return fastEvaluate(self.model.params, X, Y, padToken)

    #Returns the boolean result for each case in the data set where the data set is pre encoded
    def evaluateEncoded(self, X, Y, customName = None, doPrint = True, outputArray = True):
        self.setForwardFun()

        if doPrint:
            if customName:
                print("Evaluating model:",customName)
            else:
                print("Evaluating model:",self.name)

        N=len(X)
        booleanAccuracy = np.zeros(N)
        if outputArray:
            booleanAccuracy = np.zeros(N)
        else:
            booleanAccuracy = 0

        #Finds the padding token of model
        padToken = self.model.input_encoder.encoding_map["compiler_pad"]
        maxLength = X.shape[1]
        
        i = 0
        for x, y in zip(X, Y):
            logits = forward_fun.apply(self.model.params, jax.numpy.array([x])).unembedded_output
            pred = jnp.argmax(logits, axis=-1)[0]
            
            #Boolean accuracy on all considered tokens
            mask = jnp.ones_like(x)
            mask = mask.at[0].set(0)
            padMask = jnp.where(x!=padToken, mask, 0)

            if outputArray:
                booleanAccuracy[i] = jnp.all(pred*padMask == y*padMask)
            else:
                booleanAccuracy += jnp.all(pred*padMask == y*padMask)
            i+=1
            
        if outputArray:
            return booleanAccuracy
        else:
            return booleanAccuracy / N
    
    #Apply model to a sample
    def apply(self, input):
        return self.model.apply(input).decoded
    
    #Perform a forward pass for training
    def forward(self, x):
        self.setForwardFun()
        return forward_fun.apply(self.model.params, x)    
    
    def train(self, X_train, Y_train, n_epochs=1, batch_size=8, lr=0.0001, plot=False, X_val = None, Y_val = None, valCount = 0, valStep=0, loss_fn = cross_entropy_loss):
        trainer = Trainer(
            model=self.model,
            params=self.model.params,
            X_train=X_train,
            Y_train=Y_train,
            loss_fn=loss_fn,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            plot=plot,
            X_val=X_val,
            Y_val=Y_val,
            valCount=valCount,
            valStep=valStep
        )

        result = trainer.train()
        self.model.params = trainer.model.params

        return result

    #Grid search with the hyperparameters on random weights
    def gridSearch(self, X_train, Y_train, X_test, Y_test, n_epochs_values = [100], batch_size_values = [256], learning_rate_values = [1e-5], 
                   Gaussian_mean_values = [0], Gaussian_std_values = [1], averageCount = 5):
        startingParams = self.model.params

        self.setForwardFun()
        padToken = self.model.input_encoder.encoding_map["compiler_pad"]

        # Create an empty DataFrame to store the results
        results = []

        # Perform hyper-parameter search
        for n_epochs in n_epochs_values:
            for batch_size in batch_size_values:
                for learning_rate in learning_rate_values:
                    for Gaussian_mean in Gaussian_mean_values:
                        for Gaussian_std in Gaussian_std_values:
                            setResults = np.zeros(averageCount)
                            for i in range(averageCount):   
                                # Train the model with the current hyper-parameters
                                self.setRandomWeights(Gaussian_mean, Gaussian_std)
                                self.train(X_train, Y_train, n_epochs=n_epochs, batch_size=batch_size, lr=learning_rate)
                                
                                # Evaluate the model on the test set
                                accuracy = fastEvaluate(self.model.params, X_test, Y_test, padToken)
                                setResults[i]=accuracy
                                
                            # Append the results to the DataFrame
                            mean = np.mean(setResults)
                            std = np.std(setResults)
                            results.append({"n_epochs": n_epochs, "batch_size": batch_size, "learning_rate": learning_rate, 
                                            "Gaussian_mean": Gaussian_mean, "Gaussian_std": Gaussian_std, "accuracy_mean": mean, "accuracy_std": std})

        self.setWeights(startingParams)
        # Print the results
        print(pd.DataFrame(results).to_string())
    
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
    
