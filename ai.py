# AI for Self Driving Car

# Importing the libraries

import numpy as np #numpy is open source python library used in science and  maths  
import random #generates random floating numbers in the range of 0.1, and 1.0
import os# provides functions for interacting with the operating system
import torch#PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch library. 
import torch.nn as nn # train and build the layers of neural networks such as input, hidden, and output
import torch.nn.functional as F#modules is better choice for building networks while nn. functional are basic building blocks of those layers
import torch.optim as optim # a package implementing various optimization algorithms.
import torch.autograd as autograd #provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module): #INHERITANCE : network is child class of a larger class in nn module
   
 #THIS CLASS DEFINES EACH NEURAL NETWORK WILL CONTAIN 5 I/P NEURONS , 30 HIDDEN NEURONS & 3 O/P NEURONS
    def __init__(self, input_size, nb_action):#SELF:OBJECT TO BE CREATED ,NO. OF I/P O/P NEURONS,DIRECTIONS)
        super(Network, self).__init__() #super inherits FROM NN MODULE
        self.input_size = input_size #SPECIFY I/P SIZE
        self.nb_action = nb_action #SPECIFY NO. OF O/P NEURONS
        self.fc1 = nn.Linear(input_size, 30) #FULL CONNECTIONS B/W I/P & HIDDEN LAYER
        self.fc2 = nn.Linear(30, nb_action) #FULL CONNECTIONS B/W HIDDEN & O/P LAYER
   
    #FORWARD FN USED TO ACTIVATE THE NEURONS USING RECTIFIER ACTIVATION FUNCTION & RETURN Q-VALUES
    def forward(self, state): #SELF TO USE FC1 &FC2 , STATE AS I/P
        x = F.relu(self.fc1(state)) #ACTIVATE HIDDEN NEURONS (X) USING RELU
        q_values = self.fc2(x) #GET O/P NEURONS(Q-VALUES)
        return q_values 

# Implementing Experience Replay

class ReplayMemory(object): #
    #init fn define variables(memory) for future instance of the class 
    def __init__(self, capacity): #self:attached to future instance & capacity :100
        self.capacity = capacity#max capacity :100
        self.memory = [] #memory isnt initialized
        
        #push fn to append new event in the memory & make sure the memomry doesnt contain more than 100 transitions
    def push(self, event): #event consits of 4 elements : LAST STATE (st) , 2nd STATE(st+1), 3rd STATE(80) , LAST REWARD(rt)
        self.memory.append(event) 
        if len(self.memory) > self.capacity:#if self memory > self capacity dlt them
            del self.memory[0]
            
    #sample fn to sample some transitions in the meomry of last 100 transactions
    def sample(self, batch_size): 
        #zip fn: creates an iterator that will aggregate elements from two or more iterables / reshapes ur list
        #if list =((1,2,3),(4,5,6)), zip(*list)=((1,4),(2,3),(5,6)) : action 1 => reward 4 and like tht
        samples = zip(*random.sample(self.memory, batch_size)) #random.sample takes random samples from memory of fixed batch_size'
        #we cant reurn samples without using a pytorch variable (map) [pytorch variables create a gradient & wraps the tensor in PyTorch]
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #map() fn wraps the samples into torch varibales(lamda x) lamda fn will take the samples concatenate wrt 1st dimension & convert these tensors into torch variables containing both tensors & gradient

# Implementing Deep Q Learning

class Dqn(): 
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma #delay coefficient
        self.reward_window = [] #sliding window of mean of last 100 rewards
        self.model = Network(input_size, nb_action) #create our neural network
        self.memory = ReplayMemory(100000) #memory object
        #optimizer to perform stochastic gradient descent
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #addon has parameters : learning rate(lr)
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #vector 5 dimensions(3 directions , orientation & minus orientation) encding one state of environment
        self.last_action = 0 
        self.last_reward = 0
    
    def select_action(self, state):# STATE AS I/P(3 signals , orientation & minus orientation)
    #softmax returns probablities of 3 q values for 3 possible actions
    #1st variable : probs , 2ND VARIABLE : ACTION
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100 HIGHER THE VALUE OF T , HIGHER IS THE WINNNING PROBABILITY
        #SOFTMAX (1,2,3)=(0.04,0.11,0.85) => WHEN MULTIPLIED BY T, SOFTMAX =((1,2,3)*T) => (0,0.02,0.9) INCREASES WINNING PROBABLITY
        action = probs.multinomial(num_samples=1) #GONNA BE A RANDOM DRAW OF THE PROBABILITY DONE 
        return action.data[0,0]
    #we have batches for current state , next state , reward & action now we're using exp replay trick for the dqn to learn something
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
       #UNSQUEEZE: returns a new tensor with a dimension of size one inserted at the specified position.
       #SQUEEZE :Returns a tensor with all specified dimensions of input of size 1 removed.
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) #gather fn gathers each time the best action to play for each of the i/p states of the batch state
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #taking max of all q values in correspondance with the action
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) #temporal difference loss fn
        #backpropagate the loss fn with stochastic gradient descent to update the weights
        self.optimizer.zero_grad() #reinitialise optimiser at each iteration of the loop
        td_loss.backward(retain_graph = True) #back propagate it
        self.optimizer.step() #this will update the weights 
        #hence we have a learning deep q netwrok
    
    #update function will update everything there is to update as soon as the AI reaches a new state.
    def update(self, reward, new_signal):
        #new state consisting of new signal converted into torch tensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #torch used to convert new signal into torch tensor &type convert to 'float'
           #update memory after reaching new state 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #memory consits of last state , new state , last action & last reward. (ALL WILL BE TORCH TENSOR)
           #play an action after reaching new state & updating memory
        action = self.select_action(new_state) 
        #after playing the action , time for reward if memory more than 100 elements , time to learn
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) #create new variables taking inputs from sample tansitions
            self.learn(batch_state, batch_next_state, batch_reward, batch_action) #learning 
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward) #update reward window after getting last reward
        if len(self.reward_window) > 1000: #to limit reward window size not more than 1000 
            del self.reward_window[0] #if >1000 dlt them
        return action
    #compute score : mean of all rewards in reward window 
    def score(self): 
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    #save the brain(PROGRESS) of the car on quitting the application
    def save(self): #only saving self.model , self.optimizer tht were updated at last iteration to reuse later
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    #load it whenever we ope the appliaction
    def load(self):
        if os.path.isfile('last_brain.pth'):#os.path leads to desktop ,'isfile' searches for last_brain.pth
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth') #load the model
            self.model.load_state_dict(checkpoint['state_dict']) #update model
            self.optimizer.load_state_dict(checkpoint['optimizer']) #update optimizer
            print("done !")
        else:
            print("no checkpoint found...")
