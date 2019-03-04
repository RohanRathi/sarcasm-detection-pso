# for smaller datasets please use the simpler model sarcasm_detection_model_CNN_LSTM_DNN_simpler.py

import os
import sys

sys.path.append('../')

import collections
import time
import numpy
import keras

numpy.random.seed(1337)
from sklearn import metrics
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import defaultdict
import src.data_processing.data_handler as dh

import math
import random

basepath = os.getcwd()[:os.getcwd().rfind('/')]
train_file = basepath + '/resource/train/Train_v1.txt'
validation_file = basepath + '/resource/dev/Dev_v1.txt'
test_file = basepath + '/resource/test/Test_v1.txt'
word_file_path = basepath + '/resource/word_list_freq.txt'
split_word_path = basepath + '/resource/word_split.txt'
emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

output_file = basepath + '/resource/text_model/TestResults.txt'
model_file = basepath + '/resource/text_model/weights/'
vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

num_dimensions = 7 # 2 conv o/p and conv filter size, 2 lstm o/p and dnn o/p
                       # Can use extra dimension of dropout in lstm

class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, swarm, embedding_dimension=256, hidden_units=256, trainable=False):
        print('Build model...')
        model = Sequential()

        model.add(
            Embedding(vocab_size, embedding_dimension, input_length=maxlen, embeddings_initializer='glorot_normal'))

        model.add(Convolution1D(swarm[0], swarm[1], kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen)))
        # model.add(MaxPooling1D(pool_size=3))
        model.add(Convolution1D(swarm[2], swarm[3], kernel_initializer='he_normal', padding='valid', activation='sigmoid',
                                input_shape=(1, maxlen - 2)))
        # model.add(MaxPooling1D(pool_size=3))

        # model.add(Dropout(0.25))

        model.add(LSTM(swarm[4], kernel_initializer='he_normal', activation='sigmoid', dropout=0.5,
                       return_sequences=True))
        model.add(LSTM(swarm[5], kernel_initializer='he_normal', activation='sigmoid', dropout=0.5))

        model.add(Dense(swarm[6], kernel_initializer='he_normal', activation='sigmoid'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())
        return model


class train_model(sarcasm_model):
    train = None
    validation = None
    print("Loading resource...")

    def __init__(self, train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                 vocab_file,
                 output,
                 input_weight_file_path=None):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        self.load_train_validation_data()

        print(self._line_maxlen)

        # build vocabulary
        # truncates words with min freq=1
        self._vocab = dh.build_vocab(self.train, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)

        # prepares input
        tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        # embedding dimension
        self.dimension_size = 256

        # solving class imbalance
        self.ratio = self.calculate_label_ratio(Y)
        self.ratio = [max(self.ratio.values()) / value for key, value in self.ratio.items()]
        print('class ratio::', self.ratio)

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]
        self.X = X
        self.tX = tX
        self.Y = Y
        self.tY = tY

        print('train_X', X.shape)
        print('train_Y', Y.shape)
        print('validation_X', tX.shape)
        print('validation_Y', tY.shape)

    def build_tmp_model(self, swarm, final):
        
        # trainable true if you want word2vec weights to be updated
        # Not applicable in this code

        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, swarm, embedding_dimension=self.dimension_size,
                                    trainable=True)

        #Understand below commented I/O ops

        if final != True:
            '''open(self._model_file + 'model.json', 'w').write(model.to_json())
            save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=False)
            save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}__.hdf5',
                                       save_best_only=False)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)'''

            # training
            hist = model.fit(self.X, self.Y, batch_size=128, epochs=4, validation_data=(self.tX, self.tY), shuffle=True,
                      class_weight=self.ratio)

        else:
            open(self._model_file + 'model.json', 'w').write(model.to_json())
            save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=False)
            save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}__.hdf5',
                                       save_best_only=False)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

            # training
            hist = model.fit(self.X, self.Y, batch_size=128, epochs=4, validation_data=(self.tX, self.tY), shuffle=True,
                      callbacks=[save_best, save_all, early_stopping], class_weight=self.ratio)

        print('Rohan Rathi')
        print(model.metrics_names)
        rv = hist.history['loss'][-1]
        del model

        return rv

    def load_train_validation_data(self):
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
        print('Training data loading finished...')

        self.validation = dh.loaddata(self._validation_file, self._word_file_path, self._split_word_file_path,
                                      self._emoji_file_path,
                                      normalize_text=True,
                                      split_hashtag=True,
                                      ignore_profiles=False)
        print('Validation data loading finished...')

        if (self._test_file != None):
            self.test = dh.loaddata(self._test_file, self._word_file_path, normalize_text=True,
                                    split_hashtag=True,
                                    ignore_profiles=True)

    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels):
        return collections.Counter(labels)


class test_model(sarcasm_model):
    test = None
    model = None

    def __init__(self, model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file,
                 input_weight_file_path=None):
        print('initializing...')
        sarcasm_model.__init__(self)

        self._model_file_path = model_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._vocab_file_path = vocab_file_path
        self._output_file = output_file
        self._input_weight_file_path = input_weight_file_path

        print('test_maxlen', self._line_maxlen)

    def load_trained_model(self, model_file='model.json', weight_file='model.json.hdf5'):
        start = time.time()
        self.__load_model(self._model_file_path + model_file, self._model_file_path + weight_file)
        end = time.time()
        print('model loading time::', (end - start))

    def __load_model(self, model_path, model_weight_path):
        self.model = model_from_json(open(model_path).read())
        print('model loaded from file...')
        self.model.load_weights(model_weight_path)
        print('model weights loaded from file...')

    def load_vocab(self):
        vocab = defaultdict()
        with open(self._vocab_file_path, 'r') as f:
            for line in f.readlines():
                key, value = line.split('\t')
                vocab[key] = value

        return vocab

    def predict(self, test_file, verbose=False):
        try:
            start = time.time()
            self.test = dh.loaddata(test_file, self._word_file_path, self._split_word_file_path, self._emoji_file_path,
                                    normalize_text=True, split_hashtag=True,
                                    ignore_profiles=False)
            end = time.time()
            if (verbose == True):
                print('test resource loading time::', (end - start))

            self._vocab = self.load_vocab()
            print('vocab loaded...')

            start = time.time()
            tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.test, self._vocab)
            tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)
            end = time.time()
            if (verbose == True):
                print('test resource preparation time::', (end - start))

            self.__predict_model(tX, self.test)
        except Exception as e:
            print('Error:', e)
            raise

    def __predict_model(self, tX, test):
        y = []
        y_pred = []

        prediction_probability = self.model.predict_proba(tX, batch_size=1, verbose=1)

        try:
            fd = open(self._output_file + '.analysis', 'w')
            for i, (label) in enumerate(prediction_probability):
                gold_label = test[i][1]
                words = test[i][2]
                dimensions = test[i][3]
                context = test[i][4]
                author = test[i][5]

                predicted = numpy.argmax(prediction_probability[i])

                y.append(int(gold_label))
                y_pred.append(predicted)

                fd.write(str(label[0]) + '\t' + str(label[1]) + '\t'
                         + str(gold_label) + '\t'
                         + str(predicted) + '\t'
                         + ' '.join(words))

                fd.write('\n')

            print()

            print('accuracy::', metrics.accuracy_score(y, y_pred))
            print('precision::', metrics.precision_score(y, y_pred, average='weighted'))
            print('recall::', metrics.recall_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.f1_score(y, y_pred, average='weighted'))
            print('f_score::', metrics.classification_report(y, y_pred))
            fd.close()
        except Exception as e:
            print(e)
            raise


class particle():
    def __init__(self, init_pos, inc):
        self.w = 0.0            #Init weight as .5
        self.pos = [int]
        self.vel = []
        self.pbest = []
        self.min_err = 1
        self.err = 1

        self.pos = init_pos
        for i in range(0, num_dimensions):
            self.vel.append(random.uniform(-1*inc[i], inc[i]))

    def update_min_error(self, cur_err):
        self.err = cur_err

        if self.err < self.min_err:
            self.pbest = self.pos
            self.min_err = self.err

    def update_vel(self, gbest, inc):
    
        x = 0
        for i in range(0, num_dimensions):
            x += (self.pos[i] - self.pbest[i]) ** 2
            
        self.w = self.w + .5 * math.exp(-math.sqrt(x))

        for i in range(0, num_dimensions):
            r1 = random.random() % inc[i]        #TODO: c1, c2, r1, r2
            r2 = random.random() % inc[i]

            vel_cog = .5 * r1 * (self.pbest[i] - self.pos[i])
            vel_soc = .5 * r2 * (gbest[i] - self.pos[i])
            self.vel[i] = self.w * self.vel[i] + vel_cog + vel_soc

    def update_pos(self, max_bound, min_bound, inc):
        print('See vel here: ')
        print(self.vel)
        for i in range(0, num_dimensions):
            self.pos[i] = self.pos[i] + self.vel[i]
            
            if self.pos[i] > max_bound[i]:      #TODO: pos update
                self.pos[i] = max_bound[i]

            if self.pos[i] < min_bound[i]:
                self.pos[i] = min_bound[i]

            self.pos[i] = int(round(self.pos[i] / inc[i]) * inc[i])

#END class particle

class pso(train_model):
    maxiter = 5
    num_particles = 6

    min_bound = [32, 2, 32, 2, 64, 64, 64]
    max_bound = [256, 5, 256, 5, 256, 256, 512]
    inc = [16, 1, 16, 1, 32, 32, 32]
    gbest = None
    gbest_err = 1

    def __init__(self):

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        swarm = []

        for i in range(0, self.num_particles):
            swarm.append(particle(self.init_particle_pos(), self.inc)) 

        tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file, vocab_file_path, output_file)  

        for i in range(0, self.maxiter):
            for j in range(0, self.num_particles):      

                swarm[j].update_min_error(tr.build_tmp_model(swarm[j].pos, False))
                #print('Check len here: ')
                #print(len(tr))

                keras.backend.clear_session()
                if swarm[j].err < self.gbest_err:
                    self.gbest = swarm[j].pos
                    self.gbest_err = swarm[j].err
            
            for j in range(0, self.num_particles):
                swarm[j].update_vel(self.gbest, self.inc)
                swarm[j].update_pos(self.max_bound, self.min_bound, self.inc)

        print('END PSO!!!')
        print(swarm[0].pos)
        print(self.gbest)   
        print(self.gbest_err)
   
        tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file, vocab_file_path, output_file)
        tr.build_tmp_model(self.gbest, True)  

    def init_particle_pos(self):
        position = []
        for i in range(0, num_dimensions):
            position.append((random.randint(0, 512) % ((self.max_bound[i] - self.inc[i]) / self.inc[i])) * self.inc[i] + self.min_bound[i])
            position[-1] = int(position[-1])
        return position


if __name__ == "__main__":

    '''import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    set_session(tf.Session(config=config))'''

    # uncomment for training
    p = pso()
    '''tr = train_model(train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                     vocab_file_path, output_file)'''

    t = test_model(model_file, word_file_path, split_word_path, emoji_file_path, vocab_file_path, output_file)
    t.load_trained_model(weight_file='weights.04__.hdf5')
    t.predict(test_file)
