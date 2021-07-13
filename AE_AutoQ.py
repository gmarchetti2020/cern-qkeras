#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install  git+https://github.com/google/qkeras.git@master')


# In[1]:


import numpy as np
import h5py
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model,model_from_json
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
from tensorflow.keras.layers import ReLU, LeakyReLU
from tensorflow.keras import backend as K
from qkeras import QDense, QActivation
import math

from datetime import datetime
from tensorboard import program
import os
import pathlib
#import tensorflow_model_optimization as tfmot
#tsk = tfmot.sparsity.keras

import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

from functions import preprocess_anomaly_data, custom_loss_negative, custom_loss_training


# In[2]:


#tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()


# In[3]:


file = h5py.File('Delphes_dataset_HALF.h5', 'r')
X_train_flatten = np.array(file['X_train_flatten'])
X_test_flatten = np.array(file['X_test_flatten'])
X_val_flatten = np.array(file['X_val_flatten'])

X_train_scaled = np.array(file['X_train_scaled'])
#X_test_scaled = np.array(file['X_test_scaled'])
X_val_scaled = np.array(file['X_val_scaled'])

file.close()


# In[4]:


latent_dim = 3
input_shape = 56

#encoder
inputArray = Input(shape=(input_shape))
x = Activation('linear', name='block_1_act')(inputArray)
 #   else QActivation(f'quantized_bits(16,6,1)')(inputArray)
x = BatchNormalization(name='bn_1')(x)
x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(),use_bias=False, name='block_2_dense')(x)
x = BatchNormalization(name='bn_2')(x)
x = Activation('relu', name='block_2_act')(x)
x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(),use_bias=False, name='block_3_dense')(x)
x = BatchNormalization(name='bn_3')(x)
x = Activation('relu', name='block_3_act')(x)
encoder = Dense(latent_dim, kernel_initializer=tf.keras.initializers.HeUniform(),name='output_encoder')(x)
#x = BatchNormalization()(x)

#decoder
x = Dense(16, kernel_initializer=tf.keras.initializers.HeUniform(),use_bias=False, name='block_4_dense')(encoder)
x = BatchNormalization(name='bn_4')(x)
x = Activation('relu', name='block_4_act')(x)
x = Dense(32, kernel_initializer=tf.keras.initializers.HeUniform(),use_bias=False, name='block_5_dense')(x)
x = BatchNormalization(name='bn_5')(x)
x = Activation('relu', name='block_5_act')(x)
x = Dense(input_shape, kernel_initializer=tf.keras.initializers.HeUniform(), name='output_dense')(x)
decoder = Activation('linear', name='output_act')(x)

#create autoencoder
autoencoder = Model(inputs = inputArray, outputs=decoder)
autoencoder.summary()


# ### Load signal data

# In[5]:


ato4l = h5py.File('Ato4l_lepFilter_13TeV.h5', 'r')
ato4l = ato4l['Particles'][:]
ato4l = ato4l[:,:,:-1]

import joblib
pT_scaler = joblib.load('pt_scaled_VAE_new.dat')


# In[6]:


test_scaled_ato4l, test_notscaled_ato4l = preprocess_anomaly_data(pT_scaler, ato4l)


# ### Set objective and  compile the model

# In[7]:


bsm_data = test_notscaled_ato4l #input - data without any preprocessing
#obj = roc_objective(autoencoder, X_test_flatten[:1000], bsm_data)
autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=custom_loss_training, run_eagerly=True) # just to make sure it runs in eager


# ### Override AutoQKeras classes

# In[8]:


from Custom_AutoQKeras import *


# ### Set AutoQKeras parameters

# In[9]:


from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
import pprint


# In[10]:


physical_devices = tf.config.list_physical_devices()
for d in physical_devices:
    print(d)


# In[11]:


reference_internal = "fp32"
reference_accumulator = "fp32"

q = run_qtools.QTools(
  autoencoder,
  # energy calculation using a given process
  # "horowitz" refers to 45nm process published at
  # M. Horowitz, "1.1 Computing's energy problem (and what we can do about
  # it), "2014 IEEE International Solid-State Circuits Conference Digest of
  # Technical Papers (ISSCC), San Francisco, CA, 2014, pp. 10-14, 
  # doi: 10.1109/ISSCC.2014.6757323.
  process="horowitz",
  # quantizers for model input
  source_quantizers=[quantized_bits(16, 6, 1)],
  is_inference=False,
  # absolute path (including filename) of the model weights
  # in the future, we will attempt to optimize the power model
  # by using weight information, although it can be used to further
  # optimize QBatchNormalization.
  weights_path=None,
  # keras_quantizer to quantize weight/bias in un-quantized keras layers
  keras_quantizer=reference_internal,
  # keras_quantizer to quantize MAC in un-quantized keras layers
  keras_accumulator=reference_accumulator,
  # whether calculate baseline energy
  for_reference=True)
  
# caculate energy of the derived data type map.
energy_dict = q.pe(
    # whether to store parameters in dram, sram, or fixed
    weights_on_memory="sram",
    # store activations in dram or sram
    activations_on_memory="sram",
    # minimum sram size in number of bits. Let's assume a 16MB SRAM.
    min_sram_size=8*16*1024*1024,
    rd_wr_on_io=False)

# get stats of energy distribution in each layer
energy_profile = q.extract_energy_profile(
    qtools_settings.cfg.include_energy, energy_dict)
# extract sum of energy of each layer according to the rule specified in
# qtools_settings.cfg.include_energy
total_energy = q.extract_energy_sum(
    qtools_settings.cfg.include_energy, energy_dict)

pprint.pprint(energy_profile)
print()
print("Total energy: {:.2f} uJ".format(total_energy / 1000000.0))


# In[12]:


quantization_config = {
        "kernel": {
                "quantized_bits(2,1,1,alpha=1.0)": 2,
                "quantized_bits(4,2,1,alpha=1.0)": 4,
                "quantized_bits(6,2,1,alpha=1.0)": 6,
                "quantized_bits(8,3,1,alpha=1.0)": 8,
                "quantized_bits(10,3,1,alpha=1.0)": 10,
                "quantized_bits(12,4,1,alpha=1.0)": 12,
                "quantized_bits(14,4,1,alpha=1.0)": 14,
                "quantized_bits(16,6,1,alpha=1.0)": 16
        },
        "bias": {
                "quantized_bits(2,1,1)": 2,
                "quantized_bits(4,2,1)": 4,
                "quantized_bits(6,2,1)": 6,
                "quantized_bits(8,3,1)": 8
        },
        "activation": {
                "quantized_relu(2,1)": 2,
                "quantized_relu(3,1)": 3,
                "quantized_relu(4,2)": 4,
                "quantized_relu(6,2)": 6,
                "quantized_relu(8,3)": 8,
                "quantized_relu(10,3)": 10,
                "quantized_relu(12,4)": 12,
                "quantized_relu(14,4)": 14,
                "quantized_relu(16,6)": 16
        },
        "linear": {
                "quantized_bits(16,6)": 16
        }
}


# In[13]:


limit = {
    "Dense": [16, 8, 16],
    "Activation": [16]
}


# In[14]:


goal = {
    "type": "energy",
    "params": {
        "delta_p": 8.0,
        "delta_n": 8.0,
        "rate": 4.0, # a try
        "stress": 0.6, # a try
        "process": "horowitz",
        "parameters_on_memory": ["sram", "sram"],
        "activations_on_memory": ["sram", "sram"],
        "rd_wr_on_io": [False, False],
        "min_sram_size": [0, 0],
        "source_quantizers": ["fp16"],
        "reference_internal": "fp16",
        "reference_accumulator": "fp16"
        }
}


# In[15]:


odir='autoqkeras'


# In[16]:


run_config = {
    "output_dir": "{}/".format(odir),
    "goal": goal,
    "quantization_config": quantization_config,
    "learning_rate_optimizer": False,
    "transfer_weights": False,
    "mode": "bayesian",
    #"score_metric": "val_roc_objective_val",
    "seed": 42,
    "limit": limit,
    "tune_filters": "layer",
    "tune_filters_exceptions": "^output.*",
    "layer_indexes": [1,3,5,6,8,9,10,12,13,15,16,17],
    "max_trials": 130,
    "blocks": [
          "block_1_.*$",
          "block_2_.*$",
          "block_3_.*$",
          "output_encoder$",
          "block_4_.*$",
          "block_5_.*$",
          "output_dense$",
          "output_act$",],
    "schedule_block": "cost"
}


# In[17]:


print("quantizing layers:", [autoencoder.layers[i].name for i in run_config["layer_indexes"]])


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

callbacks=[]
#if pruning=='pruned':
 #   callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=2, min_lr=1E-6))
#callbacks.append(TerminateOnNaN())
#callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='{}/AUTOQKERAS_best.h5'.format(odir),monitor="val_loss",verbose=0,save_best_only=True))
#callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='{}/AUTOQKERAS_best_weights.h5'.format(odir),monitor="val_loss",verbose=0,save_weights_only=True))
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',verbose=1, patience=8))


# In[19]:


EPOCHS = 25
BATCH_SIZE = 1024


# ### Run search with AutoQ

# In[20]:


start = time.time()


# In[ ]:


autoqk = Custom_AutoQKerasScheduler(autoencoder,metrics=[custom_loss_negative], X_test = X_test_flatten[:3000000], bsm_data = bsm_data,                             custom_objects={}, debug=False, **run_config)
autoqk.fit(X_train_flatten[:3000000], X_train_scaled[:3000000],           validation_data=(X_val_flatten[:3000000], X_val_scaled[:3000000]),           batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks)


# In[ ]:


end = time.time()
print(end - start)


# In[ ]:


qmodel = autoqk.get_best_model()
qmodel.summary()
save_model('best_pretrain_objective_roc', qmodel)

