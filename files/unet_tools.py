#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:08:55 2020

Unet models

@author: amt
"""

import tensorflow as tf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def make_large_unet(fac, sr,ncomps=3):
    # BUILD THE MODEL
    # These models start with an input
    if sr==40:
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(600,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(600,6)) # 1 Channel seismic data
      
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #300
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #150
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #75
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(75,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((75,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        
    elif sr==100:
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(1500,6)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        
    if sr==250:
        input_layer=tf.keras.layers.Input(shape=(120,2)) # 1 Channel seismic data
      
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #60
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #30
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #15
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(15,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((15,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    return model

def make_large_unet_b(fac, sr,ncomps=3):
    # BUILD THE MODEL
    # These models start with an input
    if sr==100:
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(1500,6)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        output=tf.keras.layers.Conv1D(2,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        #output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        
    model.summary()
    
    return model

def make_large_unet_drop(fac,sr,ncomps=1):
    
    if sr==40:
        # These models start with an input
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(600,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(600,6)) # 1 Channel seismic data
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #300
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #150
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #75
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(75,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((75,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        # Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        # Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        # End of network
        network=tf.keras.layers.Dropout(.2)(level1)
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(network) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    elif sr==100:
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(1500,6)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Dropout(.2)(level1)
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    return model

def make_large_unet_drop_b(fac, sr, ncomps=1):
    
    if sr==100:
        if ncomps==1:
            input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        elif ncomps==3:
            input_layer=tf.keras.layers.Input(shape=(1500,6)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Dropout(.2)(level1)
        output=tf.keras.layers.Conv1D(2,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        #output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    return model

def my_3comp_data_generator(batch_size,x_data,n_data,sig_inds,noise_inds,sr,std,valid=False,epsilon=1e-6):
    while True:

        if valid:
            start_of_noise_batch=0
            start_of_data_batch=0
        else:
            # randomly select a starting index for the data batch
            start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
            # randomly select a starting index for the noise batch
            start_of_noise_batch=np.random.choice(len(noise_inds)-batch_size//2)
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size//2]
        # get range of indicies from noise
        noiseinds=noise_inds[start_of_noise_batch:start_of_noise_batch+batch_size//2]
        #length of each component
        nlen=int(sr*30)+1
        # grab batch
        comp1=np.concatenate((x_data[datainds,:nlen],n_data[noiseinds,:nlen]))
        comp2=np.concatenate((x_data[datainds,nlen:2*nlen],n_data[noiseinds,nlen:2*nlen]))
        comp3=np.concatenate((x_data[datainds,2*nlen:],n_data[noiseinds,2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((np.ones_like(datainds),np.zeros_like(noiseinds)))
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/nlen*np.ones((1,nlen))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)
        
def my_3comp_data_generator_nonoise(batch_size,x_data,sig_inds,sr,std,valid=False):
    epsilon=1e-6
    while True:
        # randomly select a starting index for the data batch
        start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
        if valid:
            start_of_data_batch=0
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size]
        #length of each component
        nlen=int(sr*30)+1
        # grab batch
        comp1=x_data['waves'][datainds,:nlen]
        comp2=x_data['waves'][datainds,nlen:2*nlen]
        comp3=x_data['waves'][datainds,2*nlen:]
        # make target data vector for batch
        target=np.ones_like(datainds)
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/nlen*np.ones((1,nlen))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)
        
def my_lfe_data_generator(batch_size,x_data,n_data,sig_inds,noise_inds,sr,std,valid=False):
    epsilon=1e-6
    while True:
        if valid:
            start_of_noise_batch=0
            start_of_data_batch=0
        else:
            # randomly select a starting index for the data batch
            start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
            # randomly select a starting index for the noise batch
            start_of_noise_batch=np.random.choice(len(noise_inds)-batch_size//2)
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size//2]
        # get range of indicies from noise
        noiseinds=noise_inds[start_of_noise_batch:start_of_noise_batch+batch_size//2]
        #length of each component
        nlen=int(sr*30)+1
        # grab batch
        comp1=np.concatenate((x_data['waves'][datainds,:nlen],n_data['waves'][noiseinds,:nlen]))
        comp2=np.concatenate((x_data['waves'][datainds,nlen:2*nlen],n_data['waves'][noiseinds,nlen:2*nlen]))
        comp3=np.concatenate((x_data['waves'][datainds,2*nlen:],n_data['waves'][noiseinds,2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((np.ones_like(datainds),np.zeros_like(noiseinds)))
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/nlen*np.ones((1,nlen))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)
        
def my_3comp_data_generator_b(batch_size,x_data,n_data,dts,sig_inds,noise_inds,sr,std,valid=False):
    
    epsilon=1e-6
    while True:
        # randomly select a starting index for the data batch
        if valid:
            start_of_noise_batch=0
            start_of_data_batch=0
        else:
            start_of_data_batch=np.random.choice(len(sig_inds)-batch_size//2)
            # randomly select a starting index for the noise batch
            start_of_noise_batch=np.random.choice(len(noise_inds)-batch_size//2)
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size//2]
        # get range of indicies from noise
        noiseinds=noise_inds[start_of_noise_batch:start_of_noise_batch+batch_size//2]
        #length of each component
        nlen=x_data.shape[1]//3
        # grab batch
        comp1=np.concatenate((x_data[list(datainds),:nlen],n_data[list(noiseinds),:nlen]))
        comp2=np.concatenate((x_data[list(datainds),nlen:2*nlen],n_data[list(noiseinds),nlen:2*nlen]))
        comp3=np.concatenate((x_data[list(datainds),2*nlen:],n_data[list(noiseinds),2*nlen:]))
        # make target data vector for batch
        target=np.concatenate((dts[datainds],np.zeros_like(noiseinds)))
        # make structure to hold target functions
        batch_targetp=np.zeros((batch_size,nlen))
        batch_targets=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.zeros(len(target))
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_targetp[ii,:]=np.zeros((1,nlen))
                batch_targets[ii,:]=np.zeros((1,nlen))
                time_offset[ii]=np.random.uniform(0,nlen-sr*winsize)
            elif targ>0:
                # make a gaussian
                tmp=signal.gaussian(nlen,std=int(std*sr))
                #shift and pad it to align with p
                #print(str(targ))
                batch_targetp[ii,:]=np.concatenate((tmp[int(targ*sr):],np.zeros(int(targ*sr))))
                # add p and s targets                        
                batch_targets[ii,:]=signal.gaussian(nlen,std=int(std*sr))
                #time_offset[ii]=np.random.uniform(nlen//2+1-sr*winsize,nlen//2+1)
                #time_offset[ii]=np.random.uniform(nlen//2+1-sr*winsize,nlen//2+1-sr*winsize+(sr*winsize-targ*sr))
                time_offset[ii]=np.random.uniform(nlen//2+1-targ*sr-sr*winsize,nlen//2+1)
        #print(time_offset)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_targetp=np.zeros((batch_size,int(winsize*sr))) 
        new_batch_targets=np.zeros((batch_size,int(winsize*sr))) 
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset) #HZ sampling Frequency
            start_bin=bin_offset 
            if start_bin < 0: # this
                start_bin=0
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_targetp[ii,:]=batch_targetp[ii,start_bin:end_bin]
            new_batch_targets[ii,:]=batch_targets[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        new_batch_target=[]
        for ii in range(new_batch_targetp.shape[0]):
            batch_out.append(np.hstack([new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                        new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                        new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)]))
            new_batch_target.append(np.hstack([new_batch_targetp[ii,:].reshape(-1,1), new_batch_targets[ii,:].reshape(-1,1)]))
        batch_out=np.array(batch_out)
        new_batch_target=np.array(new_batch_target)
        yield(batch_out,new_batch_target)
        
def my_3comp_obs_data_generator_b(batch_size,x_data,sig_inds,sr,std,valid=False):
    
    epsilon=1e-6
    while True:
        # randomly select a starting index for the data batch
        if valid:
            start_of_data_batch_p=0
            start_of_data_batch_s=0
        else:
            start_of_data_batch_p=np.random.choice(len(sig_inds)-batch_size)
            start_of_data_batch_s=np.random.choice(len(sig_inds)-batch_size)
        # get range of indicies from data
        dataindsp=sig_inds[start_of_data_batch_p:start_of_data_batch_p+batch_size]
        dataindss=sig_inds[start_of_data_batch_s:start_of_data_batch_s+batch_size]
        #length of each component
        nlen=x_data.shape[1]//3
        # grab batch
        #sos = signal.butter(4, 1, 'hp', fs=sr, output='sos')
        comp1p=x_data[list(dataindsp),:nlen] # get true P
        comp2p=x_data[list(dataindsp),nlen:2*nlen]
        comp3p=x_data[list(dataindsp),2*nlen:]
        comp1s=x_data[list(dataindss),:nlen] # filter the S
        comp2s=x_data[list(dataindss),nlen:2*nlen]
        comp3s=x_data[list(dataindss),2*nlen:]
        # make target data vector for batch, this is s-p in seconds
        target=np.random.uniform(1,15,size=batch_size)
        # initialize structures to hold spliced data
        comp1=np.zeros_like(comp1p)
        comp2=np.zeros_like(comp1p)
        comp3=np.zeros_like(comp1p)
        fac=10
        for ii in range(len(target)):
            # print("INDEX:"+str(ii))
            # print(comp1p[ii,:].shape)
            # print(np.zeros((int(target[ii]/2*sr))).shape)
            # print(comp1s[ii,:-int(target[ii]/2*sr)].shape)
            # print(target[ii])
            # mask out everything prior to P up to halfway between P and S
            taper=signal.tukey(int(sr*target[ii]),alpha=1)[:int(sr*target[ii])]
            smask=np.concatenate((np.zeros(nlen//2),taper,np.ones(nlen-len(taper)-nlen//2)))
            pmask=1-smask
            comp1p[ii,:]=comp1p[ii,:]*pmask
            comp2p[ii,:]=comp2p[ii,:]*pmask
            comp3p[ii,:]=comp3p[ii,:]*pmask    
            comp1s[ii,:]=comp1s[ii,:]*smask
            comp2s[ii,:]=comp2s[ii,:]*smask
            comp3s[ii,:]=comp3s[ii,:]*smask
            # plt.figure()
            # plt.plot(pmask)
            # plt.plot(smask)
            comp1[ii,:]=comp1p[ii,:]+fac*np.concatenate((np.zeros((int(target[ii]*sr))),comp1s[ii,:-int(target[ii]*sr)]))
            comp2[ii,:]=comp2p[ii,:]+fac*np.concatenate((np.zeros((int(target[ii]*sr))),comp2s[ii,:-int(target[ii]*sr)]))
            comp3[ii,:]=comp3p[ii,:]+fac*np.concatenate((np.zeros((int(target[ii]*sr))),comp3s[ii,:-int(target[ii]*sr)]))
            # plt.figure()
            # plt.plot(comp1[ii,:],label='comp1')
            # plt.plot(comp1p[ii,:],label='comp1p')
            # plt.plot(comp1s[ii,:],label='comp1s')
            # plt.legend()
            # plt.title(target[ii])
        # make structure to hold target functions
        batch_targetp=np.zeros((batch_size,nlen))
        batch_targets=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.zeros(len(target))
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            # make a gaussian
            tmp=signal.gaussian(nlen,std=int(std*sr))
            # add p and s targets                        
            # print(targ)
            batch_targetp[ii,:]=signal.gaussian(nlen,std=int(std*sr))
            batch_targets[ii,:]=np.concatenate((np.zeros(int(targ*sr)),tmp[:-int(targ*sr)]))
            #time_offset[ii]=np.random.uniform(nlen//2+1-sr*winsize,nlen//2+1)
            #time_offset[ii]=np.random.uniform(nlen//2+1-sr*winsize,nlen//2+1-sr*winsize+(sr*winsize-targ*sr))
            time_offset[ii]=np.random.randint(int(targ*sr),(sr*winsize))
            # print("s-p is "+str(targ)+" and time shift is "+str(time_offset[ii]))
            
        #print(time_offset)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_targetp=np.zeros((batch_size,int(winsize*sr))) 
        new_batch_targets=np.zeros((batch_size,int(winsize*sr))) 
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            start_bin=int(offset) 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_targetp[ii,:]=batch_targetp[ii,start_bin:end_bin]
            new_batch_targets[ii,:]=batch_targets[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        new_batch_target=[]
        for ii in range(new_batch_targetp.shape[0]):
            batch_out.append(np.hstack([new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                        new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                        new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)]))
            new_batch_target.append(np.hstack([new_batch_targetp[ii,:].reshape(-1,1), new_batch_targets[ii,:].reshape(-1,1)]))
        batch_out=np.array(batch_out)
        new_batch_target=np.array(new_batch_target)
        yield(batch_out,new_batch_target)
        
def my_3comp_obs_data_generator(batch_size,x_data,sig_inds,sr,std,valid=False):
    epsilon=1e-6
    while True:
        # randomly select a starting index for the data batch
        start_of_data_batch=np.random.choice(len(sig_inds)-batch_size)
        # randomly select a starting index for the noise batch
        if valid:
            start_of_data_batch=0
        # get range of indicies from data
        datainds=sig_inds[start_of_data_batch:start_of_data_batch+batch_size]
        #length of each component
        nlen=int(sr*30)+1
        # grab batch
        comp1=x_data[datainds,:nlen]
        comp2=x_data[datainds,nlen:2*nlen]
        comp3=x_data[datainds,2*nlen:]
        # make target data vector for batch
        target=np.ones_like(datainds)
        # make structure to hold target functions
        batch_target=np.zeros((batch_size,nlen))
        # shuffle things (not sure if this is needed)
        inds=np.arange(batch_size)
        np.random.shuffle(inds)
        comp1=comp1[inds,:]
        comp2=comp2[inds,:]
        comp3=comp3[inds,:]
        target=target[inds]
        # some params
        winsize=15 # winsize in seconds
        # this just makes a nonzero value where the pick is
        for ii, targ in enumerate(target):
            #print(ii,targ)
            if targ==0:
                batch_target[ii,:]=1/nlen*np.ones((1,nlen))
            elif targ==1:
                batch_target[ii,:]=signal.gaussian(nlen,std=int(std*sr))
        # I have 30 s of data and want to have 15 s windows in which the arrival can occur anywhere
        time_offset=np.random.uniform(0,winsize,size=batch_size)
        # initialize arrays to hold shifted data
        new_batch=np.zeros((batch_size,int(winsize*sr),3))
        new_batch_target=np.zeros((batch_size,int(winsize*sr)))        
        # this loop shifts data and targets and stores results
        for ii,offset in enumerate(time_offset):
            bin_offset=int(offset*sr) #HZ sampling Frequency
            start_bin=bin_offset 
            end_bin=start_bin+int(winsize*sr) 
            new_batch[ii,:,0]=comp1[ii,start_bin:end_bin]
            new_batch[ii,:,1]=comp2[ii,start_bin:end_bin]
            new_batch[ii,:,2]=comp3[ii,start_bin:end_bin]
            new_batch_target[ii,:]=batch_target[ii,start_bin:end_bin]
        # does feature log
        new_batch_sign=np.sign(new_batch)
        new_batch_val=np.log(np.abs(new_batch)+epsilon)
        batch_out=[]
        for ii in range(new_batch_target.shape[0]):
            batch_out.append(np.hstack( [new_batch_val[ii,:,0].reshape(-1,1), new_batch_sign[ii,:,0].reshape(-1,1), 
                                          new_batch_val[ii,:,1].reshape(-1,1), new_batch_sign[ii,:,1].reshape(-1,1),
                                          new_batch_val[ii,:,2].reshape(-1,1), new_batch_sign[ii,:,2].reshape(-1,1)] ) )
        batch_out=np.array(batch_out)
        yield(batch_out,new_batch_target)
        
