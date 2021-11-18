# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 00:57:03 2021

@author: Ameer
"""
#   Importing the Libraries
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import math
import numpy as np
import random
from collections import deque
import pandas as pd
from sklearn.utils import shuffle
import os
from tensorflow.keras.models import save_model, load_model

import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import model_from_json
#import numba as nb
#from numba import jit, cuda
#from timeit import default_timer as timer
#   Creating the Agent
#
no_of_effort = 0.1
os.getcwdb()

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 2 # fault-prone / non-fault-prone
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        #self.model = load_model(model_name) if is_eval else self._model()
        self.model = tf.keras.models.load_model("C:/Users/Ameer/Documents/sdp/data/model/columba/" + model_name + ".h5") if is_eval else self._model()
        
        
    def _model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="softmax"))
        ## Define multiple optional optimizers
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)
        
        ## Compile model with metrics
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
                
        return model
    
    #@jit
    def act(self, state): 
        if not self.is_eval and random.random()<= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        print(options)
        return np.argmax(options[0])
    
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
  
#            
#   Define basic function
#
def getDataVec(key):
    vec = []
    #lines = open("C:/Users/Ameer/Documents/sdp/data/"+key+".csv","r").read().splitlines()
    vec = pd.read_csv("C:/Users/Ameer/Documents/sdp/data/"+key+".csv")
    l = int((len(vec) + 1)*no_of_effort)
    vec.head(l)
    #vec.drop('bug', axis=1, inplace=True)
    return vec 

def sigmoid(x):
    return 1/(1+math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    res = []
    for i in range(n - 1):
        res.append(data.iloc[d + 1])
    return np.array([res])


def getSize(key):
    vec = []
    lines = open("C:/Users/Ameer/Documents/sdp/data/"+key+".csv","r").read().splitlines()
    l = int((len(lines) + 1)*no_of_effort)
    lines[:l]
    for line in lines[1:]:
        #print(line)
        #print(float(line.split(",")[4]))
        vec.append(float(line.split(",")[6]))
        #print(vec)
    return vec
    
def getAction(key):
    vec = []
    lines = open("C:/Users/Ameer/Documents/sdp/data/"+key+".csv","r").read().splitlines()
    l = int((len(lines) + 1)*no_of_effort)
    lines[:l]
    for line in lines[1:]:
        #print(line)
        #print(float(line.split(",")[4]))
        vec.append(float(line.split(",")[14]))
        #print(vec)
    return vec

#
#   Training the Agent
#
import sys
file_name = str("columba_v1")
window_size = int(1)
episode_count = int(10)

agent = Agent(window_size)
data = getDataVec(file_name)
bug = data.count(1)
no_bug = data.count(0)

state = getState(data, 0, window_size + 1)
size = getSize(file_name)
prone = getAction(file_name)

overall_predicted_state = []
overall_effort_epoch = []
overall_benefit = []
overall_cost = []
overall_defective = []
overall_non_defective = []
overall_benefit_all = []
overall_cost_all = []
overall_benefit = []
overall_cost = []

summation_state = []

l = int((len(data) - 1)*no_of_effort)
batch_size = 32

for e in range(episode_count):
    #start = timer()
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    
    
    effort = 0
    overall_effort = 0
    non_defective = 0
    defective = 0
    benefit = 0
    cost = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
            
        next_state = getState(data, t + 1, window_size + 1) 
        effort = 0
        reward = 0
        
        summation_state.append(int(action))
        overall_predicted_state.append(action)
        
        if action  == 0: # non-fault prone
            #agent.inventory.append(data[t])
            
            if action == prone[t]:
                reward = 1
            else:
                reward = -1
                
            non_defective = non_defective + 1
            
            print(defective)
            print(non_defective)
            print("Non-fault prone: " + str(action))
            
        elif action == 1 and len(agent.inventory) > 0: # fault prone
            if action  == prone[t]:
                reward = 1
            else:
                reward = -1
                
                
            defective = defective + 1
            
            print(defective)
            print(non_defective)
            print("Fault Prone: " + str(action))
        
        #   Calculate effort
        if size[t] != 0 and defective != 0 and reward == 1:
            effort = action / (defective/size[t])
            list_effort = agent.inventory.append(effort)
            print ("Effort to detect defect: " + str(effort))
        else:
            effort = 0
            list_effort = agent.inventory.append(effort)
            print ("Effort to detect defect: " + str(effort))
            
        overall_effort = overall_effort + effort
        
        #   Calculate benefit --> pindah kt done
        benefit = defective * overall_predicted_state[t]
        overall_benefit.append(benefit)
        
        #   Calculate cost
        cost = defective * effort
        overall_cost.append(cost)
        
        
        done = True if t == l - 1 else False
        
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        
        
        if done:
            overall_benefit_all = sum(overall_benefit)
            overall_cost_all = sum(overall_cost)
            print("--------------------------------")
            print("---- TRAINING OF THE MODEL -----")
            print("Overall Effort in the Dataset: " + str(overall_effort))  
            print("Overall Defect in the Dataset: " + str(defective))
            print("Overall Non-Defect in the Dataset: " + str(non_defective))
            print("Overall Benefit: " + str(overall_benefit_all))
            print("Overall Cost: " + str(overall_cost_all))
            print("--------------------------------")
            overall_effort_epoch.append(overall_effort)
            overall_defective.append(defective)
            overall_non_defective.append(non_defective)
            
            
            mydata = pd.DataFrame(overall_effort_epoch, columns=['effort'])
            mydata['defective'] = overall_defective
            mydata['non-defective'] = overall_non_defective
            mydata['benefit'] = overall_benefit_all
            mydata['cost'] = overall_cost_all
            #mydata['benefit'] = overall_benefit
            #mydata['cost'] = overall_cost
            mydata.to_csv('C:/Users/Ameer/Documents/sdp/data/result/columba/result_columba_100_10ep.csv',index=False)
            
            mydata2 = pd.DataFrame(overall_benefit, columns=['benefit'])
            mydata2['cost'] = overall_cost
            mydata2.to_csv('C:/Users/Ameer/Documents/sdp/data/commit/columba/result_columba_10_10ep_episode_'+ str(e) +'.csv',index=False)
            overall_benefit = [] 
            overall_cost = []
            agent.model.save("C:/Users/Ameer/Documents/sdp/data/model/columba/model_10_ep" +str(e)+".h5")
            # serialize model to JSON
        model_json=agent.model.to_json()
        with open("C:/Users/Ameer/Documents/sdp/data/model/columba/model_10_ep" +str(e)+".json", "w") as json_file:
            json_file.write(model_json)
            
            #nb.cuda.profile_stop()
            #print("Time:", timer()-start)
        
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
        
         
#   Evaluation of the Model
#
print("--------------------------------")
print("--- EVALUATION OF THE MODEL  ---")
print("--------------------------------")
file_name = str("columba_v1")
model_name = str("model_10_ep0")

window_size = int(1)

agent = Agent(window_size, True, model_name)
data = getDataVec(file_name)
bug = data.count(1)
no_bug = data.count(0)

no_of_effort = 1

l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
size = getSize(file_name)
prone = getAction(file_name)

overall_predicted_state = []
overall_effort_epoch = []
overall_benefit = []
overall_cost = []
overall_defective = []
overall_non_defective = []
overall_benefit_all = []
overall_cost_all = []
overall_benefit = []
overall_cost = []

summation_state = []

effort = 0
overall_effort = 0
non_defective = 0
defective = 0
benefit = 0
cost = 0
agent.inventory = []

for t in range(l):
    action = agent.act(state)
        
    next_state = getState(data, t + 1, window_size + 1) 
    effort = 0
    reward = 0
    
    summation_state.append(int(action))
    overall_predicted_state.append(action)
    
    if action  == 0: # non-fault prone
        #agent.inventory.append(data[t])
        
        if action == prone[t]:
            reward = 1
        else:
            reward = -1
            
        non_defective = non_defective + 1
        
        print(defective)
        print(non_defective)
        print("Non-fault prone: " + str(action))
        
    elif action == 1 and len(agent.inventory) > 0: # fault prone
        if action  == prone[t]:
            reward = 1
        else:
            reward = -1
            
            
        defective = defective + 1
        
        print(defective)
        print(non_defective)
        print("Fault Prone: " + str(action))
    
    #   Calculate effort
    if size[t] != 0 and defective != 0:
        effort = action / (defective/size[t])
        list_effort = agent.inventory.append(effort)
        print ("Effort to detect defect: " + str(effort))
    else:
        effort = 0
        list_effort = agent.inventory.append(effort)
        print ("Effort to detect defect: " + str(effort))
        
    overall_effort = overall_effort + effort
    
    #   Calculate benefit --> pindah kt done
    benefit = defective * overall_predicted_state[t]
    overall_benefit.append(benefit)
    
    #   Calculate cost
    cost = defective * effort
    overall_cost.append(cost)
    
    
    done = True if t == l - 1 else False
    
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
          
    if done:
        overall_benefit_all = sum(overall_benefit)
        overall_cost_all = sum(overall_cost)
        print("--------------------------------")
        print("---- TRAINING OF THE MODEL -----")
        print("Overall Effort in the Dataset: " + str(overall_effort))  
        print("Overall Defect in the Dataset: " + str(defective))
        print("Overall Non-Defect in the Dataset: " + str(non_defective))
        print("Overall Benefit: " + str(overall_benefit_all))
        print("Overall Cost: " + str(overall_cost_all))
        print("--------------------------------")
        overall_effort_epoch.append(overall_effort)
        overall_defective.append(defective)
        overall_non_defective.append(non_defective)
        
        
        mydata = pd.DataFrame(overall_effort_epoch, columns=['effort'])
        mydata['defective'] = overall_defective
        mydata['non-defective'] = overall_non_defective
        mydata['benefit'] = overall_benefit_all
        mydata['cost'] = overall_cost_all
        #mydata['benefit'] = overall_benefit
        #mydata['cost'] = overall_cost
        mydata.to_csv('C:/Users/Ameer/Documents/sdp/data/result/columba/result_evaluation_columba_10_ep2.csv',index=False)
        
        mydata2 = pd.DataFrame(overall_benefit, columns=['benefit'])
        mydata2['cost'] = overall_cost
        mydata2.to_csv('C:/Users/Ameer/Documents/sdp/data/commit/columba/result_evaluation_columba_10_ep2.csv',index=False)
        overall_benefit = [] 
        overall_cost = []