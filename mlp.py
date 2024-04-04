import numpy as np
from scipy.special import softmax #using predifined softmax function

def expand(Y): #Expantion of the actual output
    ExY=np.zeros((Y.size, Y.max()+1))
    ExY[np.arange(Y.size), Y] = 1
    ExY=ExY.T
    return ExY

class MLP():

    # Constructor Function. Defines the paremeters of the MLP model
    def __init__(self, layers, act_funct, weight, learning_rate):
        '''
        Arguments:
        layers:[Input, Hidden, Hidden, Output]
        act_funct:n Activation Function. Sigmoid and ReLU functions are the choices
        weight: Assumption of the starting weight. Either gaussian weights or zeros
        learning_rate: As it names
        '''

        self.layers = layers
        self.n_layers = len(layers)
        self.act_f = act_funct
        self.wt = weight
        self.rate = learning_rate
        
        #initalizing weights
        self.initialize_weights()


    def sigmoid(self, z):
        # Sigmoid function. z can be scalar or vector
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def relu(self, z):
        # ReLU function. Leaky ReLU is being used here to avoid divide by zero error.
        # z can be scalar or vector
        return np.maximum(z, 0.01*z)

    def sigmoid_derivative(self, z):
        # Derivative for Sigmoid function
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result

    def relu_derivative(self, z):
        # Derivative for ReLU function
        dx = np.ones_like(z)
        dx[z < 0] = 0.01
        return dx

    def actf(self, z):
        # Selection of the activation fuction depending on the arguments.

        if self.act_f=='relu':
            # print("Applied ReLU")
            result=self.relu(z)
        elif self.act_f=='sigmoid':
            result=self.sigmoid(z)
        return result

    def actf_dz(self, z):
        # Selection of the derivative of the activation fuction depending on the arguments.

        if self.act_f=='relu':
            result=self.relu_derivative(z)
        elif self.act_f=='sigmoid':
            result=self.sigmoid_derivative(z)
        return result
    
    

    def initialize_weights(self):
        #Weight initialization. Selection of the starting weight depending on the arguments.

        self.weights = []
        self.bias=[]
        next_layers = self.layers.copy()
        next_layers.pop(0)
        for curr_layer, next_layer in zip(self.layers, next_layers):
            if self.wt == 'normal':
                # if self.act_f=='relu':
                #     mod=np.sqrt(2/curr_layer)
                # else:
                #     mod=np.sqrt(2/(curr_layer+next_layer))
                weight = np.random.normal(0,1, size=(curr_layer, next_layer))
                bias = np.ones((next_layer, 1))
            elif self.wt == 'zeros':
                weight=np.zeros((curr_layer, next_layer))
                bias = np.ones((next_layer, 1))
            self.weights.append(weight.T)
            self.bias.append(bias)
        return self.weights, self.bias

    def train(self, X, Y):
        # For each epoc 
        self.forward(X)
        self.gradients, self.inter = self.backward(X, Y)
        # self.weights, self.bias = 
        self.update_wt()

    def forward(self, X):
        # Forward pass
        input_layer = X
        # Defining the intermediate latent valiables
        self.A = [None] * self.n_layers
        self.Z = [None] * self.n_layers
        #Calculation for the forward pass
        self.A[0]=input_layer
        self.Z[1]=self.weights[0].dot(self.A[0])+self.bias[0]
        self.A[1]=self.actf(self.Z[1])
        self.Z[2]=self.weights[1].dot(self.A[1])+self.bias[1]
        self.A[2]=self.actf(self.Z[2])
        self.Z[3]=self.weights[2].dot(self.A[2])+self.bias[2]
        self.A[3]=softmax(self.Z[3], axis=0)

        return self.A, self.Z

    def backward(self, X, Y):
        # Backpropagation

        n_examples = X.shape[0]
        ExY=expand(Y)   #Expanding Y from 1D to 2D
        dels = [None] * self.n_layers
        grads=[None]*(self.n_layers-1)
        db=[None]*(self.n_layers-1)

        dels[3]=(self.A[3]-ExY)
        grads[2]=dels[3].dot(self.A[2].T)/n_examples
        db[2]=np.sum(dels[3],1)/n_examples

        dels[2]=self.weights[2].T.dot(dels[3])*self.actf_dz(self.Z[2])
        grads[1]=dels[2].dot(self.A[1].T)/n_examples
        db[1]=np.sum(dels[2],1)/n_examples

        dels[1]=self.weights[1].T.dot(dels[2])*self.actf_dz(self.Z[1])
        grads[0]=dels[1].dot(self.A[0].T)/n_examples
        db[0]=np.sum(dels[1],1)/n_examples

        return grads, db



    def update_wt(self):
        # Updating Weights

        self.weights[0]=self.weights[0]-self.rate*self.gradients[0]
        self.weights[1]=self.weights[1]-self.rate*self.gradients[1]
        self.weights[2]=self.weights[2]-self.rate*self.gradients[2]

        self.bias[0]=self.bias[0]-self.rate*np.reshape(self.inter[0], (25,1))
        self.bias[1]=self.bias[1]-self.rate*np.reshape(self.inter[1], (10,1))
        self.bias[2]=self.bias[2]-self.rate*np.reshape(self.inter[2], (10,1))

        return self.weights, self.bias


    def predict(self):
       return np.argmax(self.A[3], axis=0)

    def accuracy(self, P,Y):
        return np.sum(P==Y)/Y.shape

    def loss(self, Y):
        #Loss calculation
        ExY=expand(Y)
        result=0.5*np.square(self.A[3]-ExY).mean()
        return result