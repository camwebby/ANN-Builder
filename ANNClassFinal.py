import numpy as np
from matplotlib import pyplot as plt

###test data####
trainFeatures1 = np.array( [[0,0,1],
                                       [0,1,1],
                                       [1,0,1],
                                       [1,1,1],
                                       [0,0,0]] )

trainLabels1 = np.array([[0,1,1,0,0]]).T
### end of test data###


##Example network with 3 layers, the first having 6 neurons, second having 2, third having 3:
layersAndWeights = np.array([[6],[2],[3]])

##neurons and weights form: [[x],[y],[z]],
##where dimesions of array show how many layers and value in each
##shows the weights per layer
##One can have as many layers as one desires, e.g. [[x]] for one layer, or [[],[],[],[],[],[],[],........[]] 


##defining the class
class ANN():

##one can specify the following variables when creating the network:
    def __init__(self, trainFeatures, trainLabels, dimensions, activation, iterations):

        self.trainFeatures = trainFeatures
        self.trainLabels = trainLabels
        self.dimensions = dimensions
        self.activation = activation
        self.iterations = int(iterations)


    def initialiseWeights(self):
        np.random.seed(1)
        
        self.weights = []

        #creating the first weight
        trainFeatColumns = int(self.trainFeatures.shape[1])
        firstLayerSynapses = int(self.dimensions[0])
        
        self.weights.append(2*np.random.random((trainFeatColumns,firstLayerSynapses)) - 1)

        #creating middle weights

        for count,synapses in enumerate(self.dimensions):
            if count < len(self.dimensions) - 1:
                x = int(self.dimensions[count])
                y = int(self.dimensions[count+1])
                self.weights.append(2*np.random.random((x,y)) - 1)
            else:
                break

        #creating the last neuron to output(s) weight(s)
        trainLabelColumns = int(self.trainLabels.shape[1])
        lastWeight = self.weights[len(self.weights)-1].shape
        self.weights.append(2*np.random.random((lastWeight[1],trainLabelColumns)) - 1)


    def activationFunc(self, func, x, deriv=False):
        if func == "tanh":
            if deriv == True:
                return 1/np.cosh(x**2)
            return np.tanh(x)
            
        elif func== "sigmoid":
            if deriv == True:
                return np.exp(-x)/((1+np.exp(-x))**2)
            return 1/(1+np.exp(-x))
        else:
            print("mistyped or unknown activation function. try 'sigmoid' or 'tanh'")
                

    def forwardProp(self):
        X = self.trainFeatures
        self.zn = [X]
        self.an = [X]

        for count,weigh in enumerate(self.weights):
            dotProd = np.dot(self.zn[count], self.weights[count])
            self.zn.append(self.activationFunc(self.activation, dotProd))
            self.an.append(dotProd)
            
        
    def getError(self):
        return self.trainLabels - self.zn[len(self.zn) - 1]
        
    def backProp(self):
        error = self.getError()
        
        self.deltas = []
        self.pdifferentials = []
        
        #finding partial differential of error with respect to the last weight(s)
        deltan = np.multiply(-(error), self.activationFunc(self.activation, self.an[len(self.an) - 1], True))
        dEdwn = np.dot(self.zn[len(self.zn) - 2].T, deltan)
        self.deltas.append(deltan)
        self.pdifferentials.append(dEdwn)
        
        #finding the partial differential of error with respect to the remenaining weight(s)
        for count, idk in enumerate(self.zn):
            if count < len(self.zn)-2:
                self.deltas.append(np.dot(self.deltas[count], self.weights[len(self.weights) - (count+1)].T)*self.activationFunc(self.activation, self.an[len(self.an) - (count+2)] , True))
                self.pdifferentials.append( np.dot( self.zn[len(self.zn) - (count+3)].T, self.deltas[count+1]))
            else:
                break
          
    def updateWeights(self):

        for count, (diff, weigh) in enumerate(zip(self.pdifferentials, self.weights)):
            self.weights[count] -= self.pdifferentials[len(self.pdifferentials)-(count+1)] 


    def run(self, printError=False, costGraph=False):

        self.averageCost = []
        self.interval = []

        self.initialiseWeights()
        
        for iter in range(self.iterations):
            self.forwardProp()
            self.getError()
            self.backProp()
            self.updateWeights()
            if costGraph==True:
                if iter % int(self.iterations*0.01) == 0:
                    self.averageCost.append(abs(np.mean(self.getError())))
                    self.interval.append(iter)
           
        if printError == True:
          print(self.getError())

        if costGraph==True:           
            plt.plot(neuralNet1.interval, neuralNet1.averageCost)
            plt.show()


if __name__ == "__main__":
    neuralNet1 = ANN(trainFeatures=trainFeatures1, trainLabels=trainLabels1, dimensions=layersAndWeights, activation="sigmoid", iterations=2000)
    neuralNet1.run(printError=True,costGraph=True)



