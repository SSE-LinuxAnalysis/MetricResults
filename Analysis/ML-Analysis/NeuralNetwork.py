#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

# Only supported/required in Colab: https://stackoverflow.com/a/58803943
# %tensorflow_version 2.x

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model as Model_
from tensorflow.keras.metrics import *

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *

import time
import numpy as np

class NeuralNet(Model_):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.denseLayers  = []

    self.denseLayers.append(Dense(units=512, activation='relu', kernel_regularizer='l2'))
    self.denseLayers.append(Dense(units=512, activation='relu', kernel_regularizer='l2'))
    self.denseLayers.append(Dense(units=1, activation='sigmoid', kernel_regularizer='l2'))

  def call(self, input_x, training=True):

    # Pass through the outputs of each conv layer to the next one
    output = input_x
    for layer in self.denseLayers:
      if layer.name == 'dropout':
        output = layer(output, training)
      else: 
        output = layer(output)

    return output
    
    
class Optimizer:
  def __init__(self, model, mb = 128, lr = 0.0001):
    self.model     = model
    self.loss      = tf.keras.losses.BinaryCrossentropy()
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    self.mb        = mb

    self.train_loss     = tf.keras.metrics.Mean(name='train_loss')

    self.test_loss      = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy  = tf.keras.metrics.BinaryAccuracy(name='test_accu')
    self.test_prec      = tf.keras.metrics.Precision(name='test_prec')
    self.test_rec       = tf.keras.metrics.Recall(name='test_rec')
    self.test_AUC       = tf.keras.metrics.AUC(name="test_auc")
  
  @tf.function
  def train_step(self, x , y):
    with tf.GradientTape() as tape:
      
      predictions = self.model(x, training=True)
      loss = self.loss(y, predictions)
    
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
    y = tf.reshape(y,(-1,1))
  
    self.train_loss.update_state(loss)

  @tf.function
  def test_step(self, x , y):
    predictions = self.model(x, training=False)
    loss = self.loss(y, predictions)
    
    y, predictions = tf.reshape(y,[-1,]), tf.reshape(predictions, [-1,])

    self.test_AUC.update_state(y, predictions)
    self.test_loss.update_state(loss)
    self.test_accuracy.update_state(y, predictions)
    self.test_prec.update_state(y, predictions)
    self.test_rec.update_state(y, predictions)
    
  def train (self):
    for mbX, mbY in self.train_ds:
      self.train_step(mbX, mbY)
  
  def test  (self):
    for mbX, mbY in self.test_ds:
      self.test_step(mbX, mbY)  
  
  def reset_states(self):
      # Reset the loss and acc values after each epoch
      self.train_loss.reset_states()
      self.test_loss.reset_states()
      self.test_accuracy.reset_states()
      self.test_prec.reset_states()
      self.test_rec.reset_states()
      self.test_AUC.reset_states()

  def print_output(self, verbose, epoch):
    print_interval = 10

    print_template_verbose2 = "EPOCH: {}, TRAIN Loss: {}, TEST Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}, AUC: {}"
    print_template_verbose3 = "EPOCH: {}, TRAIN Loss: {}, TEST Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}, AUC: {}, "

    # Verbose=0 doesnt print anything at all
    if verbose == 0:
      # Do nothing
      dummy = None

    # Verbose=1 prints no train F1 and only every print_interval epochs
    elif verbose == 1:
      if (epoch + 1) % print_interval == 0 or epoch == 0:
        print(print_template_verbose2.format(epoch+1,     
                  round(float(self.train_loss.result()),4),
                  round(float(self.test_loss.result()),4),
                  round(float(self.test_accuracy.result()*100), 3),
                  round(float(self.test_prec.result()),3),
                  round(float(self.test_rec.result()),3),
                  round(float(self.test_f1),9),
                  round(float(self.test_AUC.result()),4)
                  ))

    # Verbose=2 prints no train F1 and every epoch with duration
    elif verbose == 2:
      print(print_template_verbose2.format(epoch+1,     
                round(float(self.train_loss.result()),4),
                round(float(self.test_loss.result()),4),
                round(float(self.test_accuracy.result()*100), 3),
                round(float(self.test_prec.result()),3),
                round(float(self.test_rec.result()),3),
                round(float(self.test_f1),3),
                round(float(self.test_AUC.result()),4)
                ))
      print("Duration of last epoch was: {} sec.".format(round((time.time() - self.start_epoch), 4)))
      print("----------------------------------------------------------------------")
          
    # Verbose=3 print with train_f1, every epoch, with duration
    elif verbose == 3:
      print(print_template_verbose3.format(epoch+1,     
                round(float(self.train_loss.result()),4),
                round(float(self.test_loss.result()),4),
                round(float(self.test_accuracy.result()*100), 3),
                round(float(self.test_prec.result()),3),
                round(float(self.test_rec.result()),3),
                round(float(self.test_f1),3),
                round(float(self.test_AUC.result()),4)
                ))
      print("Duration of last epoch was: {} sec.".format(round((time.time() - self.start_epoch), 4)))
      print("----------------------------------------------------------------------")

  def store_metrics(self, epoch, verbose):
    if verbose < 3:
      # If first epoch create the numpy array
      if epoch == 0:
        self.history = np.array([[epoch,
                                self.train_loss.result(),
                                0, # Because verbose < 3 doesnt store train f1
                                self.test_loss.result(),
                                self.test_accuracy.result(),
                                self.test_prec.result(),
                                self.test_rec.result(),
                                self.test_f1,
                                self.test_AUC.result()
        ]])

      # If not first epoch concatenate
      else:
        self.history = np.concatenate([self.history, [[
                                                      epoch,
                                                      self.train_loss.result(),
                                                      0, # Because verbose < 3 doesnt store train f1
                                                      self.test_loss.result(),
                                                      self.test_accuracy.result(),
                                                      self.test_prec.result(),
                                                      self.test_rec.result(),
                                                      self.test_f1,
                                                      self.test_AUC.result()
        ]]], axis=0)
    else:
      # If first epoch create the numpy array
      if epoch == 0:
        self.history = np.array([[epoch,
                                self.train_loss.result(),
                                #self.train_f1.result(),
                                0,
                                self.test_loss.result(),
                                self.test_accuracy.result(),
                                self.test_prec.result(),
                                self.test_rec.result(),
                                self.test_f1,
                                self.test_AUC.result()
        ]])

      # If not first epoch concatenate
      else:
        self.history = np.concatenate([self.history, [[
                                                      epoch,
                                                      self.train_loss.result(),
                                                      0,
                                                      self.test_loss.result(),
                                                      self.test_accuracy.result(),
                                                      self.test_prec.result(),
                                                      self.test_rec.result(),
                                                      self.test_f1,
                                                      self.test_AUC.result()
        ]]], axis=0)

  def run   (self, dataX, dataY, testX, testY, epochs, verbose=1, logging=False):
    
    # Timestamps used to compute duration of training.
    start_total = time.time()
   
    # Preparing train and test data set
    print("Loading data set...")
    self.train_ds = tf.data.Dataset.from_tensor_slices((dataX, dataY)).shuffle(dataX.shape[0]).batch(self.mb)
    self.test_ds  = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(4000)
    print("Data set loaded.")
    
    for epoch in range(epochs):
      if verbose > 0:
        self.start_epoch = time.time()

      self.train()
      self.test()
      
      self.test_f1 = 2 * ( (self.test_prec.result() * self.test_rec.result())/(self.test_prec.result() + self.test_rec.result()))
      self.print_output(verbose, epoch)
      
      self.store_metrics(epoch, verbose)

      self.reset_states()

    self.history[np.isnan(self.history)] = -1
    best_test_f1_idx = np.nanargmax(self.history[:,7])
    print("###################################################################")
    print("#                            Summary                              #")
    print("###################################################################")
    print('# Total duration of training was {} sec.'.format(round((time.time()-start_total)), 1))
    print("# Best Training Loss: " + str(np.amin(self.history[:,1])))
    print("# Best Training F1-Score: " + str(np.amin(self.history[:,2])))
    print("# Best Testing Loss: " + str(np.amin(self.history[:,3])))
    print("# Best Testing Accuracy: " + str(np.amax(self.history[:,4])))
    print("# Best Testing F1's Precision: " + str(self.history[best_test_f1_idx,5]))
    print("# Best Testing F1's Recall: " + str(self.history[best_test_f1_idx,6]))
    print("# Best Testing F1-Score: " + str(self.history[best_test_f1_idx,7]))
    print("# Best Testing AUC-ROC: " + str(np.amax(self.history[:,8])))
    print("###################################################################")
    print("# Best Testing Precision Overall: " + str(np.amax(self.history[:,5])))
    print("# Best Testing Recall Overall: " + str(np.amax(self.history[:,6])))
    
    return np.amax(self.history[:,4]), self.history[best_test_f1_idx,5], self.history[best_test_f1_idx,6], self.history[best_test_f1_idx,7], np.amax(self.history[:,8])
    # if verbose > 0:
      # return self.history
    # else:
      # return np.amax(self.history[:,4]), self.history[best_test_f1_idx,5], self.history[best_test_f1_idx,6], self.history[best_test_f1_idx,7], np.amax(self.history[:,8])