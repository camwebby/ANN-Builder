import numpy as np
import matplotlib.pyplot as plt

###test data####
trainFeatures1 = np.array( [[0,0,1],
                                       [0,1,1],
                                       [1,0,1],
                                       [1,1,1],
                                       [0,0,0]] )

trainLabels1 = np.array([[0,1,1,0,0]]).T

layersAndWeights = np.array([[6], [2], [3]])

##neurons and weights form: [[x],[y],[z]],
##where dimesions of array show how many layers and value in each
##shows the weights per layer


class ANN():


    randomValues = 2*np.random.random((3,6)) - 1
    def __init__(self, trainFeatures, trainLabels, dimensions, activation, iterations):

        self.trainFeatures = trainFeatures
        self.trainLabels = trainLabels
        self.dimensions = dimensions
        self.activation = activation
        self.iterations = int(iterations)


    def defineWeights(self):
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


    def forwardProp(self):
        X = self.trainFeatures
        z2 = np.dot(X, self.weights[0])
        lol = []

        for count,weigh in enumerate(self.weights):
            print(weigh)
        
                

            

neuralNet1 = ANN(trainFeatures1,trainLabels1,layersAndWeights,"sigmoid",1000)
neuralNet1.defineWeights()
neuralNet1.forwardProp()


    
