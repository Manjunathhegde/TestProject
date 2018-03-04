import os
import sys
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib
from math import ceil
import math
from textblob import TextBlob as tb
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import random
from collections import Counter
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models.keyedvectors import KeyedVectors

# word_vectors = KeyedVectors.load_word2vec_format('/home/manjunath/images/entity-centric-kb-pop/python/vectors.bin', binary=True)
lemmatizer = WordNetLemmatizer()
stopwords = open('stopwords.txt','r').readlines()

class PredictorClass(object):
    """ basic object model """

    def __init__(self,sequence_length=10,data_type=np.float32):
        self.sequence_length = sequence_length;
        self.data_type = data_type;
        self.text = [];
        self.corpus_length = 0;
        self.num_examples_train = 0;
        self.num_examples_test = 0;
        self.feature_length = 200;
        self.char_indices = dict;
        self.indices_char = dict;
        self.sentences_enc = []
        self.sentences_dec = []
        self.sentences_enc_test = []
        self.sentences_dec_test = []
        
    def ReadFile(self,filename_enc='train.enc',filename_dec='train.dec',file_type = 'train'): #default small data file
        lines_enc = open(filename_enc,"r",encoding="latin-1").readlines()[:20000]
        lines_dec = open(filename_dec,"r",encoding="latin-1").readlines()[:20000]
        print('--',len(lines_enc))
        print('--',len(lines_dec))
        for i, lenc in enumerate(lines_enc):
            ldec = lines_dec[i]
            if(len(lenc.split()) > 5 and len(lenc.split()) < 10 and len(ldec.split()) > 5 and len(ldec.split()) < 10):
                if(file_type == 'train'):
                    self.sentences_enc.append(lenc) #10 word max length sentence (so t=10)
                    self.sentences_dec.append(ldec)   #y is decoder sentence
                else:
                    self.sentences_enc_test.append(lenc) #40 char length sentence (so t=40)
                    self.sentences_dec_test.append(ldec)   #y is 41 character    

        self.num_examples_train = len(self.sentences_enc)
        self.num_examples_test = len(self.sentences_enc_test)
        
        print('Read complete')
        print('Train set length:', len(self.sentences_enc))
        print('Test set length:', len(self.sentences_enc_test))
        
        #get set of words from the doc
        
         
        #create training data in the following format
        #Each training sample is can contain max of 10 words. convert the sentence into x1 to x10 
        #each is a 200 dim glove vector
        
    def PrepareInputData(self, batch_size, batch_count,input_type):    
        #create input tensors
        # each input (x_t) is a one hot vector
        start = (batch_count*batch_size)
        end = start+batch_size
        if(input_type == 'train'):
            if(start + batch_size > self.num_examples_train):
                end = self.num_examples_train
                batch_size = end-start
        else:
            if(start + batch_size > self.num_examples_test):
                end = self.num_examples_test
                batch_size = end-start

        # define numpy arrays
        X_np = np.zeros((batch_size, self.sequence_length, self.feature_length), dtype=self.data_type)
        y_np = np.zeros((batch_size, self.sequence_length, self.feature_length), dtype=self.data_type)

        for i in range(start,end,1):
            if(input_type == 'train'):
                X_np[i-start] = sentence_to_vector(self.sentences_enc[i])
                y_np[i-start] = sentence_to_vector(self.sentences_dec[i])
            else:
                X_np[i-start] = sentence_to_vector(self.sentences_enc_test[i])
                y_np[i-start] = sentence_to_vector(self.sentences_dec_test[i])
        
        return X_np,y_np,batch_size
    
    def PrepareTestData(self, text):
        test = np.zeros((1,self.sequence_length,self.feature_length))
        test [0] = sentence_to_vector(text)
        return test

    def sentence_to_vector(sentence):
        featureset = []
        current_words = word_tokenize(sentence.lower())
        features = np.zeros(200)
        count = 0
        for word in current_words:
            vec = np.zeros(200)
            try:
                vec = word_vectors.word_vec(word)
                featureset.append(vec)
                count += 1
            except:
                pass
        for i in range(count,self.sequence_length):
            vec = np.zeros(200)
            featureset.append(vec)
        return featureset

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    '''Create the encoding layer'''
    lstm = rnn.BasicLSTMCell(rnn_size)
    drop = rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    enc_cell = rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_cell,
                                                   cell_bw = enc_cell,
                                                   sequence_length = sequence_length,
                                                   inputs = rnn_inputs, 
                                                   dtype=tf.float32)
    return enc_state

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''
    
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
        
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, 
                                                                vocab_size, 
                                                                None, 
                                                                scope=decoding_scope,
                                                                weights_initializer = weights,
                                                                biases_initializer = biases)

        train_logits = decoding_layer_train(encoder_state, 
                                            dec_cell, 
                                            dec_embed_input, 
                                            sequence_length, 
                                            decoding_scope, 
                                            output_fn, 
                                            keep_prob, 
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state, 
                                            dec_cell, 
                                            dec_embeddings, 
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'], 
                                            sequence_length - 1, 
                                            vocab_size,
                                            decoding_scope, 
                                            output_fn, keep_prob, 
                                            batch_size)

    return train_logits, infer_logits

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
sequence_length = 10
predict_object = PredictorClass(sequence_length)
predict_object.ReadFile('train.enc','train.dec','train')
predict_object.ReadFile('test.enc','test.dec','test')

x = tf.placeholder('float', [None, predict_object.sequence_length, predict_object.feature_length]) #time_stamp*num_feature
y = tf.placeholder('float', [None, predict_object.sequence_length, predict_object.feature_length])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# test_output = open('test_output.txt','w')
# train_neural_network(predict_object,x)
# test_output.close()
