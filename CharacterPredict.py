import os
import sys
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib
from math import ceil
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

class PredictorClass(object):
    """ basic object model """

    def __init__(self,sequence_length,steps,data_type=np.float32):
        self.sequence_length = sequence_length;
        self.steps = steps;
        self.data_type = data_type;
        self.text = '';
        self.corpus_length = 0;
        self.num_examples = 0;
        self.feature_length = 0;
        self.char_indices = dict;
        self.indices_char = dict;
        self.sentences = []
        self.next_chars = []
        self.sentences_test = []
        self.next_chars_test = []
        
    def ReadFile(self,filename='ptb.train1.txt'): #default small data file
        self.text = open(filename).read().lower()[:100000]
        print('Read complet - corpus length:', len(self.text))
        
        #get set of chars from the doc
        chars = sorted(list(set(self.text)))
        self.corpus_length = len(self.text)
        self.feature_length = len(chars)
        print('unique chars: {}'.format(self.feature_length))

        #create index dictionaries
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        
        #create training data in the following format
        #Each training sample is 40 character length. Predict the 41st character
        for i in range(0, self.corpus_length - self.sequence_length, self.steps):
            if(np.random.randint(10)<8):
                self.sentences.append(self.text[i: i + self.sequence_length]) #40 char length sentence (so t=40)
                self.next_chars.append(self.text[i + self.sequence_length])   #y is 41 character
            else:
                self.sentences_test.append(self.text[i: i + self.sequence_length]) #40 char length sentence (so t=40)
                self.next_chars_test.append(self.text[i + self.sequence_length])   #y is 41 character
        self.num_examples = len(self.sentences)
        print('num training examples: {}'.format(self.num_examples))
        print('num testing examples: {}'.format(len(self.sentences_test)))

    def PrepareInputData(self, batch_size, batch_count):    
        #create input tensors
        # each input (x_t) is a one hot vector
        start = (batch_count*batch_size)
        end = start+batch_size
        if(start + batch_size > self.num_examples):
            end = self.num_examples
            batch_size = end-start

        # define numpy arrays
        X_np = np.zeros((batch_size, self.sequence_length, self.feature_length), dtype=self.data_type)
        y_np = np.zeros((batch_size, self.feature_length), dtype=self.data_type)

        for i in range(start,end,1):
            sentence = self.sentences[i]
            for t, char in enumerate(sentence):
                X_np[i-start, t, self.char_indices[char]] = 1
            y_np[i-start, self.char_indices[self.next_chars[i]]] = 1

        #final tensor inputs
        # x_input_tensor = tf.convert_to_tensor(X_np, name="xval", dtype=tf.float32)
        # y_input_tensor = tf.convert_to_tensor(y_np, name="yval", dtype=tf.float32)

        return X_np,y_np,batch_size
    
    def PrepareTestData(self, text):
        test = np.zeros((1,self.sequence_length,self.feature_length))
        for i,c in enumerate(text):
            test[0,i,self.char_indices[c]] = 1
        return test

def recurrent_neural_network(predict_object,x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,predict_object.feature_length])),
             'biases':tf.Variable(tf.random_normal([predict_object.feature_length]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, predict_object.feature_length])
    x = tf.split(x, predict_object.sequence_length, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']
    
    return output


def train_neural_network(predict_object,x):
    prediction = recurrent_neural_network(predict_object,x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)   
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for b_s in range(ceil(predict_object.num_examples/batch_size)):
                
                epoch_x,epoch_y,batch_size_modified = predict_object.PrepareInputData(batch_size,b_s)
                epoch_x = epoch_x.reshape((batch_size_modified, predict_object.sequence_length, predict_object.feature_length))
                epoch_y = epoch_y.reshape((batch_size_modified, predict_object.feature_length))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        #iterate over test data
        total = 0
        correct = 0
        for i,test_item_x in enumerate(predict_object.sentences_test):
            test_item_y = predict_object.next_chars_test[i]
            xval = predict_object.PrepareTestData(test_item_x)
            nc = tf.argmax(prediction, 1)         
            ind = nc.eval({x:xval.reshape((1,predict_object.sequence_length,predict_object.feature_length))}) 
            test_output.write(test_item_x+'-'+test_item_y+'-'+predict_object.indices_char[ind[0]]+'\n')
            total += 1
            if(test_item_y == predict_object.indices_char[ind[0]]):
                correct += 1
            
        print(total,'-',correct,'-',correct/total)        
        # print(test_item_x,'-',test_item_y,'-',predict_object.indices_char[ind[0]])
                    
hm_epochs = 5
batch_size = 128
rnn_size = 64
SEQUENCE_LENGTH = 40
step_size = 3
predict_object = PredictorClass(SEQUENCE_LENGTH,step_size)
predict_object.ReadFile('ptb.train.txt')

x = tf.placeholder('float', [None, predict_object.sequence_length, predict_object.feature_length]) #time_stamp*num_feature
y = tf.placeholder('float',[None, predict_object.feature_length])

test_output = open('test_output.txt','w')
train_neural_network(predict_object,x)
test_output.close()

# def GetNextData():   
#     #read file content
#     path = 'ptb.train.txt' #replace this with any small data file
#     text = open(path).read().lower()
#     print('corpus length:', len(text))

#     #get set of chars from the doc
#     chars = sorted(list(set(text)))

#     #create index dictionaries
#     char_indices = dict((c, i) for i, c in enumerate(chars))
#     indices_char = dict((i, c) for i, c in enumerate(chars))

#     print('unique chars: {}'.format(len(chars)))
#     sentences = []
#     next_chars = []
#     #create training data in the following format
#     #Each training sample is 40 character length. Predict the 41st character
#     for i in range(0, len(text) - SEQUENCE_LENGTH, step):
#         sentences.append(text[i: i + SEQUENCE_LENGTH]) #40 char length sentence (so t=40)
#         next_chars.append(text[i + SEQUENCE_LENGTH])   #y is 41 character
#     print('num training examples: {}'.format(len(sentences)))
#     # print(sentences[10])
#     # print(next_chars[10])


#     # define numpy arrays
#     X_np = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.float32)
#     y_np = np.zeros((len(sentences), len(chars)), dtype=np.float32)

#     char_len = len(chars)
#     #create input tensors
#     # each input (x_t) is a one hot vector
#     for i, sentence in enumerate(sentences):
#         for t, char in enumerate(sentence):
#             X_np[i, t, char_indices[char]] = 1
#         y_np[i, char_indices[next_chars[i]]] = 1

#     #final tensor inputs
#     # x_input_tensor = tf.convert_to_tensor(X_np, name="xval", dtype=tf.float32)
#     # y_input_tensor = tf.convert_to_tensor(y_np, name="yval", dtype=tf.float32)

#     return X_np,y_np,char_indices,indices_char
