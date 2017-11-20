#!/usr/bin/python3

from __future__ import print_function, division

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import shutil
import os
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
rng = pd.date_range(start = '2000', periods = 209, freq = 'm')
ts = pd.Series(np.random.uniform(-10,10, size = len(rng)), rng).cumsum()
ts.plot(c = 'b', title = 'Example time series')
plt.show()
print(ts.head(10))
print(ts.tail(10))
print(ts.shape)
print("============= START =================================")
TS = np.array(ts)

print(TS.shape)
num_periods = 20
f_horizon = 3 #1
print(len(TS)%num_periods)
x_data = TS[:(len(TS)-(len(TS)%num_periods))]
print(x_data.shape)
x_batches = x_data.reshape(-1,20,1)
print(x_batches.shape)
##y_data = TS[1:(len(TS)-(len(TS)%num_periods)) + f_horizon]
y_data = TS[f_horizon:(len(TS)-(len(TS)%num_periods))+f_horizon]
print(y_data.shape)
y_batches = y_data.reshape(-1,20,1)
print(len(x_batches))
print(y_batches.shape)
print(x_batches.shape)
##print(x_batches[0:2])
print("==============")

##print(y_batches[0:1])
df = TS[-(num_periods + f_horizon):]
print(df.shape)
df1 = df[:num_periods]
print(df1.shape)
df1 = df1.reshape(-1,20,1)
print(df1.shape)
df2 = TS[-num_periods:]
print(df2.shape)
df2 = df2.reshape(-1,20,1)
print(df2.shape)

def test_data(series, forecast, num_periods):
    test_x_setup = series[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1,20,1)
    testY = series[-(num_periods):].reshape(-1,20,1)
    return testX, testY

X_test, Y_test = test_data(TS,f_horizon,num_periods)

print(X_test.shape)
print(Y_test.shape)
##print(X_test)

tf.reset_default_graph()
num_periods = 20
inputs = 1
hidden = 100
output = 1

x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

print(x.shape)
print(y.shape)
print(x)

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden, activation = tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype = tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
#stacked_outputs = tf.layers.dense(stacked_rnn_output, output)

stacked_outputs = tf.layers.dense(stacked_rnn_output, hidden)
stacked_outputs = tf.reshape(stacked_outputs, [-1,hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, hidden)
stacked_outputs = tf.reshape(stacked_outputs, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_outputs, output)

outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])
loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 300

with tf.Session() as sess:
     init.run()
     for ep in range(epochs):
         sess.run(training_op, feed_dict = {x:x_batches, y:y_batches})
         if ep%100 ==0:
            mse = loss.eval(feed_dict = {x:x_batches, y:y_batches})
            print(ep,"\tMSE:",mse)
     y_pred = sess.run(outputs, feed_dict = {x:X_test})
     print(y_pred)

print("-=====================================================")
print(Y_test)
print(y_pred - Y_test)


print("============ Application ===============================")
d1 = pd.read_csv('ENDEX2.csv')
forward = 5
future = 5
lag = 15
batch_size =  15

X = np.array(d1)
X = X[len(X)-500:,1]
print(X.shape)

def to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(lag + 1 -i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis =1)
    df.dropna(inplace =  True)
    return df

df = to_supervised(X, lag)
df = np.array(df)
print(df.shape)
n1 = len(df) - forward
train, test = df[:n1,:], df[n1:,:]
X_train, y_train = train[:,:-1], train[:,-1]
print(X_train.shape)

x_batches = X_train.reshape(-1,batch_size, lag)

y_batches = y_train.reshape(-1, batch_size,1)
testX, y_test = test[:,:-1], test[:,-1]
print(testX.shape)
print(y_test.shape)
testX2 = np.ones((lag,lag),dtype = float)
testX2[:future,:]= testX
testX2 = testX2.reshape(-1,lag,lag)

X_test = np.ones((forward+future-3, lag),dtype = float)
X_test[0,:] = df[(n1-1),1:]
X_test = X_test.reshape(-1, forward+future-3,lag)
#X_test[0,:,:] = df[(n1-1),1:]
print(X_test.shape)

tf.reset_default_graph()

inputs = lag
hidden = 100
output = 1

x = tf.placeholder(tf.float32, [None, batch_size,lag])
y = tf.placeholder(tf.float32, [None, batch_size, 1])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden, activation = tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype = tf.float32)

learning_rate = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
#stacked_outputs = tf.layers.dense(stacked_rnn_output, output)

stacked_outputs = tf.layers.dense(stacked_rnn_output, hidden)
stacked_outputs2 = tf.reshape(stacked_outputs, [-1, hidden])
stacked_outputs3 = tf.layers.dense(stacked_outputs2, hidden)
stacked_outputs4 = tf.reshape(stacked_outputs3, [-1,hidden])
stacked_outputs5 = tf.layers.dense(stacked_outputs4, output)
outputs = tf.reshape(stacked_outputs5, [-1, batch_size, output])

#outputs = tf.reshape(stacked_outputs, [-1, batch_size, output])
loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 200
y2_pred = list()

with tf.Session() as sess:
     init.run()
     for ep in range(epochs):
         sess.run(training_op, feed_dict = {x:x_batches, y:y_batches})
         if ep%100 == 0:
            mse = loss.eval(feed_dict = {x:x_batches, y:y_batches})
            print(ep,"\tMSE",mse)
     y1_pred = sess.run(outputs, feed_dict = {x:testX2})
     #y2_pred.append(sess.run(outputs, feed_dict = {x:X_test[0,:,:]}))
     print(y1_pred)

print("==================================")
y_test = y_test.reshape(len(y_test),1)
print(y_test)
y1_pred = y1_pred.reshape(y1_pred.shape[1],1)
#print(y1_pred - y_test)


################# PART 1 ###########################################
print("================== PART 1 ===================================")

num_epochs = 3
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)


X,y = generateData()
print(X.shape)
print(y.shape)
print(X[:,0:5])
print(np.arange(2))
print(range(2))

tf.reset_default_graph()

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

print("====================")
print(batchX_placeholder.shape)
print(inputs_series)


# Forward pass
current_state = init_state
states_series = []
for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state


logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]

total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)


def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)





with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()

'''

##################### PART2 ########################################
print("================= PART 2 - Application ===================================")
d1 = pd.read_csv('ENDEX2.csv')

num_epochs = 5
total_series_length = 500
truncated_backprop_length = 10
state_size = 4
num_classes = 1
echo_step = 5
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
print(num_batches)

future = 5
X = np.array(d1)
X = X[len(X)-505:,1]
print(X.shape)

X1 = np.array((1,len(X)-echo_step), dtype = float)
X1 = X[:(len(X)-echo_step)]
print(X1.shape)
X2 = np.array((1,len(X)-echo_step), dtype = float)
X2 = X[echo_step:]
print(X2.shape)
X = X1
y = X2

x = X.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
y = y.reshape((batch_size, -1))

print(x.shape)
print(y.shape)
print(num_batches)


'''
forward = 5
future = 5
lag = 15
batch_size = forward #15

inputs = lag
hidden = 100
output = 1

num_epochs = 10
truncated_backprop_length = lag
state_size = hidden
num_classes = output

num_batches = 97


X = np.array(d1)
X = X[len(X)-500:,1]
print(X.shape)

def to_supervised(data, lag):
    df = pd.DataFrame(data)
    columns = [df.shift(lag + 1 -i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis =1)
    df.dropna(inplace =  True)
    return df

df = to_supervised(X, lag)
df = np.array(df)
print(df[0:10,:])
print(df[len(df)-10:len(df),:])
print(df.shape)

n1 = len(df) - forward
print(n1)

train, test = df[:n1,:], df[n1:,:]
X_train, y_train = train[:,:-1], train[:,-1]
print(X_train.shape)
print(y_train.shape)

x_batches = X_train.reshape(-1,batch_size, lag)
print(x_batches.shape)
y_batches = y_train.reshape(-1, batch_size,1)
print(y_batches.shape)

testX, y_test = test[:,:-1], test[:,-1]
testX_batches = testX.reshape(-1,batch_size,lag)
y_test_batches = y_test.reshape(-1,batch_size,1)
print(testX.shape)
print(y_test.shape)
print(testX_batches.shape)
print(y_test_batches.shape)

##testX2 = np.ones((lag,lag),dtype = float)
##print(testX2.shape)
##testX2[:forward,:]= testX
##testX2_batches = testX2.reshape(-1,batch_size,lag)
##print(testX2_batches.shape)

X_test = np.ones((forward+future, lag),dtype = float)
print(X_test.shape)
X_test[0,:] = df[(n1-1),1:]
X_test_batches = X_test.reshape(-1, batch_size,lag)
print(X_test_batches.shape)

'''


tf.reset_default_graph()

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
#batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)

b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1) # axis 1
labels_series = tf.unstack(batchY_placeholder, axis=1) # axis 1

print("====================")
print(batchX_placeholder.shape)
print(batchY_placeholder.shape)
##print(inputs_series.shape)
#print(labels_series.shape)
print(init_state.shape)
print(inputs_series)
print("===============")
print(inputs_series[0])

# Forward pass
current_state = init_state
states_series = []

for current_input in inputs_series:
    current_input = inputs_series[0]
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
    states_series.append(next_state)
    current_state = next_state


'''
current_input = inputs_series[0]
print(current_input)
current_input = tf.reshape(current_input, [batch_size, 1])
input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns
next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
states_series.append(next_state)
current_state = next_state

print(current_input)
print(input_and_state_concatenated)
print(next_state)
print(states_series)
print(current_state)

print("++++++++++++++++++++++++++++++++++")
current_input = inputs_series[1]
print(current_input)
current_input = tf.reshape(current_input, [batch_size, 1])
input_and_state_concatenated = tf.concat([current_input, current_state],1)  # Increasing number of columns

next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
states_series.append(next_state)
current_state = next_state

print(current_input)
print(input_and_state_concatenated)
print(next_state)
print(states_series)
print(current_state)
'''


logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
print(logits_series)

print("++++++++++++++++++++")
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
print(predictions_series)

print("++++++++++++++++++++")
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
print(losses)

print("++++++++++++++++++++")
total_loss = tf.reduce_mean(losses)
print(total_loss)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
print(train_step)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

x = x
y = y
_current_state = np.zeros((batch_size, state_size))
#_current_state = tf.zeros([batch_size, state_size], tf.float32) 

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        ##x,y = generateData()
        #x = x
        #y = y
        #_current_state = np.zeros((batch_size, state_size))
        #_current_state = tf.zeros(tf.float32, [batch_size, state_size])
        #_cur#rent_state = _current_state.reshape(-1,batch_size, state_size)
        #_current_state = tf.Variable((np.zeros(-1, batch_size,state_size)), dtype=tf.float32)
        
        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()



