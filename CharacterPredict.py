import os
import sys
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#read file content
path = 'data/ptb/ptb.train1.txt' #replace this with any small data file
text = open(path).read().lower()
print('corpus length:', len(text))

#get set of chars from the doc
chars = sorted(list(set(text)))

#create index dictionaries
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('unique chars: {}'.format(len(chars)))

#create training data in the following format
#Each training sample is 40 character length. Predict the 41st character
SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH]) #40 char length sentence (so t=40)
    next_chars.append(text[i + SEQUENCE_LENGTH])   #y is 41 character
print('num training examples: {}'.format(len(sentences)))
print(sentences[10])
print(next_chars[10])


# define numpy arrays
X_np = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y_np = np.zeros((len(sentences), len(chars)), dtype=np.bool)

#create input tensors
# each input (x_t) is a one hot vector
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X_np[i, t, char_indices[char]] = 1
    y_np[i, char_indices[next_chars[i]]] = 1

#final tensor inputs
x_input_tensor = tf.convert_to_tensor(X_np, name="xval", dtype=tf.bool)
y = tf.convert_to_tensor(y_np, name="yval", dtype=tf.bool)


hm_epochs = 2
num_examples = len(sentences)
n_classes = len(chars)
batch_size = 1
rnn_size = 64


def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    # preparing input
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    print(outputs[-1].shape)

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)   
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(num_examples)):
                epoch_x = x_input_tensor[i]
                epoch_y = y[i]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)                                                             ), y:mnist.test.labels}))

#train_neural_network(x_ten)
