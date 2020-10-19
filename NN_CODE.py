pip install tensorflow==1.14

import tensorflow as tf

#Importing the other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

#importing the dataset
dataset=pd.read_csv("NaClExtended.csv",encoding='unicode_escape')
X_t=dataset.iloc[:,0:2].values 
Y_t=dataset.iloc[:,8:9].values 

X_t=X_t.transpose()
Y_t=Y_t.transpose()


#Creating the placeholders
def create_placeholders():
    X=tf.placeholder(tf.float32,[2,108],name="X")
    Y=tf.placeholder(tf.float32,[1,108],name="Y")
    return X,Y



def initialization():
    with tf.variable_scope("oc", reuse=tf.AUTO_REUSE) as scope:
      W1 = tf.get_variable("W1",[5,2],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    with tf.variable_scope("oc", reuse=tf.AUTO_REUSE) as scope:
      W2 = tf.get_variable("W2",[1,5],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    with tf.variable_scope("oc", reuse=tf.AUTO_REUSE) as scope:
      b1 = tf.get_variable("b1",[5,1],initializer = tf.zeros_initializer())
    with tf.variable_scope("oc", reuse=tf.AUTO_REUSE) as scope:
      b2 = tf.get_variable("b2",[1,1],initializer = tf.zeros_initializer())
    parameters={"W1":W1,"b1":b1,"b2":b2,"W2":W2}
    return parameters

def forward_propagation(X,parameters):    
    W1=parameters["W1"]
    W2=parameters["W2"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    Z1 = tf.add(tf.matmul(W1,X),b1)                        # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.tanh(Z1)                     # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                       # Z2 = np.dot(W2, A1) + b2
    return Z2

def com_cost(Y,X,parameters,lambd):
    Z2=forward_propagation(X,parameters)
    W1=parameters["W1"]
    W2=parameters["W2"]
    L2_regularization_cost =tf.reduce_mean(np.sum(np.square(W1*lambd),keepdims=True))+tf.reduce_mean(np.sum(np.square(W2*lambd),keepdims=True))    
    cost = tf.reduce_mean(tf.abs(Z2-Y))
    actual_cost=cost+L2_regularization_cost
    return actual_cost,Z2,parameters
    return cost,Z2,parameters

def model(X_t,Y_t,lambd): 
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3
    
    parameters=initialization()
    X,Y=create_placeholders()
    p=parameters
    cost,Z2,p=com_cost(Y,X,p,lambd)
    costs=[]
    with tf.variable_scope("oc", reuse=tf.AUTO_REUSE) as scope:
      optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100000):
            seed=seed+1
            _ , temp_cost= sess.run([optimizer, cost], feed_dict={X:X_t, Y:Y_t})
            if(i==99999):
                Z2= sess.run([Z2], feed_dict={X:X_t, Y:Y_t}) 
                p= sess.run(p)
                
        
        return p,Z2
    

dataset1=pd.read_csv("CsClHydrophilicExtended.csv",encoding='unicode_escape')
X1=dataset1.iloc[:,0:2].values 
Y1=dataset1.iloc[:,8:9].values

X1=X1.transpose()
Y1=Y1.transpose()
root=[]
px=[]

min=10000


for i in range (1,25):
  p,Z2=model(X_t,Y_t,0.08*i)
  X=tf.placeholder(tf.float32,[2,10],name="X")
  Z1 = tf.add(tf.matmul(p["W1"],X),p["b1"])
  A1 = tf.keras.activations.tanh(Z1)                                    # A1 = relu(Z1)
  Z2 = tf.add(tf.matmul(p["W2"],A1),p["b2"])
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    A9=sess.run([Z2],feed_dict={X:X1})
  np_array=np.asarray(A9)
  reshaped_array = np_array.reshape(1,10)
  rmse = sqrt(mean_squared_error(Y1,reshaped_array))
  if(rmse<min):
    min=rmse
    position=0.08*i
    final_array=reshaped_array
  print(rmse,0.08*i) 
  root.append(rmse)
  px.append(0.08*i)

print(root)
print(min)

print(position)


X=tf.placeholder(tf.float32,[2,10],name="X")
Z1 = tf.add(tf.matmul(p["W1"],X),p["b1"])
A1 = tf.keras.activations.tanh(Z1)                                    # A1 = relu(Z1)
Z2 = tf.add(tf.matmul(p["W2"],A1),p["b2"]) 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    A9=sess.run([Z2],feed_dict={X:X1})
np_array=np.asarray(A9)
reshaped_array = np_array.reshape(1,10)
rmse = sqrt(mean_squared_error(Y1,reshaped_array))

print(rmse)

print(reshaped_array)

print(Y1)
