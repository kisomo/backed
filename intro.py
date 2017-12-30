#!/usr/bin/python2

from __future__ import print_function, division

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import shutil
import os
import time
import matplotlib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA

import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

from tensorflow.contrib.learn.python.learn import learn_runner

random.seed(111)

'''
def get_times(maximum_time):

    device_times = { "/cpu:0":[], "/cpu:0":[] }
    matrix_sizes = range(500,50000,50)

    for size in matrix_sizes:
        for device_name in device_times.keys():

            print("####### Calculating on the " + device_name + " #######")

            shape = (size,size)
            data_type = tf.float16
            with tf.device(device_name):
                r1 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                r2 = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype=data_type)
                dot_operation = tf.matmul(r2, r1)


            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
                    start_time = time.time()
                    result = session.run(dot_operation)
                    time_taken = time.time() - start_time
                    print(result)
                    device_times[device_name].append(time_taken)

            print(device_times)

            if time_taken > maximum_time:
                return device_times, matrix_sizes


device_times, matrix_sizes = get_times(1.5)
gpu_times = device_times["/cpu:0"]
cpu_times = device_times["/cpu:0"]
print("======================================================")
print(gpu_times)
print(cpu_times)


plt.plot(matrix_sizes[:len(gpu_times)], gpu_times, 'o-')
plt.plot(matrix_sizes[:len(cpu_times)], cpu_times, 'o-')
plt.ylabel('Time')
plt.xlabel('Matrix size')
plt.show()

'''

'''
print("++++++++++++++++++++ Example +++++++++++++++++++++++++++++++++++=")
##from sklearn import datasets, linear_model

# Import the needed libraries
#import urllib.request as request  
from urllib2 import urlopen

# Download dataset
IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']  
train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)  
test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)

# Train and test input data
Xtrain = train.drop("species", axis=1)  
Xtest = test.drop("species", axis=1)
print(Xtrain.shape)
print(Xtest.shape)
# Encode target values into binary ('one-hot' style) representation
ytrain = pd.get_dummies(train.species)  
ytest = pd.get_dummies(test.species)  
print(ytrain.shape)
print(ytest.shape)


# Create and train a tensorflow model of a neural network
def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(120, 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(120, 3), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 3), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    #print("deltas " + deltas)
    loss = tf.reduce_sum(deltas)
    #print("loss " + loss)
    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    #print("optimizer " + optimizer)
    train = optimizer.minimize(loss)
    #print("train " + train)
    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain.as_matrix(), y: ytrain.as_matrix()}))
        weights1 = sess.run(W1)
        #print(weights1)
        weights2 = sess.run(W2)
        #print(weights2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

    # Run the training for 3 different network architectures: (4-5-3) (4-10-3) (4-20-3)

loss_plot = {5: [], 10: [], 20: []} 
create_train_model(5, 300)


# Plot the loss function over iterations
num_hidden_nodes = [5, 10, 20]  
loss_plot = {5: [], 10: [], 20: []}  
weights1 = {5: None, 10: None, 20: None}  
weights2 = {5: None, 10: None, 20: None}  
num_iters = 2000


plt.figure(figsize=(12,8))  
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
plt.show()


# Evaluate models on the test set
X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y')

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest.as_matrix())]
    accuracy = 100 * sum(correct) / len(correct)
    print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))

'''

print("++++++++++++++++++++ Example testing +++++++++++++++++++++++++++++++++++=")
##from sklearn import datasets, linear_model

# Import the needed libraries
#import urllib.request as request  
from urllib2 import urlopen

# Download dataset
IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"  
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']  
train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)  
test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)

# Train and test input data
Xtrain = train.drop("species", axis=1)  
Xtest = test.drop("species", axis=1)
print(Xtrain.shape)
print(Xtest.shape)
# Encode target values into binary ('one-hot' style) representation
ytrain = pd.get_dummies(train.species)  
ytest = pd.get_dummies(test.species)  
print(ytrain.shape)
print(ytest.shape)
#print( train.head(5))
#print(ytest)


# Create and train a tensorflow model of a neural network
def create_train_model(hidden_nodes, num_iters):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(120, 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(120, 3), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, hidden_nodes), dtype=tf.float64)
    W3 = tf.Variable(np.random.rand(hidden_nodes, 120), dtype=tf.float64)
    W4 = tf.Variable(np.random.rand(120, hidden_nodes), dtype=tf.float64)
    W5 = tf.Variable(np.random.rand(hidden_nodes, 3), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    A2 = tf.sigmoid(tf.matmul(A1, W2))
    A3 = tf.sigmoid(tf.matmul(A2, W3))
    A4 = tf.sigmoid(tf.matmul(A3, W4))
    y_est = tf.sigmoid(tf.matmul(A4, W5))

    # Define a loss function
    deltas = tf.square(y_est - y)
    #print("deltas " + deltas)
    loss = tf.reduce_sum(deltas)
    #print("loss " + loss)
    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    #print("optimizer " + optimizer)
    train = optimizer.minimize(loss)
    #print("train " + train)
    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        #loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain.as_matrix(), y: ytrain.as_matrix()}))
        loss_plot.append(sess.run(loss, feed_dict={X: Xtrain.as_matrix(), y: ytrain.as_matrix()}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)
        weights3 = sess.run(W3)
        weights4 = sess.run(W4)
        #print(weights1)
        weights5 = sess.run(W5)
        #print(weights2)

    #print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[-1]))
    #print(loss_plot)
    sess.close()
    return weights1, weights2, weights3, weights4, weights5

    # Run the training for 3 different network architectures: (4-5-3) (4-10-3) (4-20-3)

#loss_plot = []  #{5: [], 10: [], 20: []} 
#create_train_model(100, 2000)
#create_train_model(100, 2500)
#create_train_model(200, 3000)


# Plot the loss function over iterations
num_hidden_nodes = 100 #[5, 10, 20]  
loss_plot = [] #{5: [], 10: [], 20: []}  
weights1 = [] #{5: None, 10: None, 20: None}  
weights2 = [] #{5: None, 10: None, 20: None}  
weights3 = [] #{5: None, 10: None, 20: None} 
weights4 = [] #{5: None, 10: None, 20: None} 
weights5 = [] #{5: None, 10: None, 20: None} 
num_iters = 3000


#plt.figure(figsize=(12,8))  
#for hidden_nodes in num_hidden_nodes:
hidden_nodes = 100
weights1, weights2, weights3, weights4, weights5 = create_train_model(hidden_nodes, num_iters)
plt.plot(range(num_iters), loss_plot, label="nn: 4-%d-%d-120-%d-3?" % (hidden_nodes,  hidden_nodes, hidden_nodes))

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12)  
plt.show()


# Evaluate models on the test set
X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')  
y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y')

#for hidden_nodes in num_hidden_nodes:

    # Forward propagation
W1 = tf.Variable(weights1)
W2 = tf.Variable(weights2)
W3 = tf.Variable(weights3)
W4 = tf.Variable(weights4)
W5 = tf.Variable(weights5)
A1 = tf.sigmoid(tf.matmul(X, W1))
A2 = tf.sigmoid(tf.matmul(A1, W2))
A3 = tf.sigmoid(tf.matmul(A2, W3))
A4 = tf.sigmoid(tf.matmul(A3, W4))
y_est = tf.sigmoid(tf.matmul(A4, W5))
print(y_est)

    # Calculate the predicted outputs
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y_est_np = sess.run(y_est, feed_dict={X: Xtest}) #, y: ytest})

print(y_est_np - ytest)


    # Calculate the prediction accuracy
correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
            for estimate, target in zip(y_est_np, ytest.as_matrix())]
accuracy = 100 * sum(correct) / len(correct)
print(len(correct))
print(sum(correct))
print("accuracy = %.2f%%" %accuracy)
#print(correct)
##print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))
print("Network architecture = 4-%d-%d-120-%d-3, accuracy = %.2f%%" % (hidden_nodes,  hidden_nodes, hidden_nodes,accuracy))
