# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:17:36 2019

@author: mwa
"""
import tensorflow as tf
import numpy as np
import tensorlayer as tl

def cal_para(var_list):
    total_var = 0
    for var in var_list:
        shape = var.get_shape()
        var_p = 1
        for dim in shape:
            var_p *= dim.value
        total_var += var_p
    return total_var

def weighted_cross_entropy(logits, labels, num_classes=4, head=None):
    with tf.name_scope('losses_seg'):
        logits = tf.cast(tf.reshape(logits, (-1, num_classes)), tf.float32)
        labels = tf.cast(tf.reshape(labels, [-1]), tf.int64)
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
        loss = tf.reduce_mean(softmax, name='entropy_mean')
    return loss

def weighted_cross_entropy_ss(logits,ss1 ,ss2,labels, num_classes=4):
     with tf.name_scope('weighted_cross_entropy'):
         logits = tf.cast(tf.reshape(logits, (-1, num_classes)), tf.float32)
         ss1 = tf.cast(tf.reshape(ss1, (-1, num_classes)), tf.float32)
         ss2 = tf.cast(tf.reshape(ss2, (-1, num_classes)), tf.float32)
         labels = tf.cast(tf.reshape(labels, [-1]), tf.int64)
         labels = tf.one_hot(labels,depth=num_classes,axis=-1)
         class_weights = tf.constant([[1.0,1.2,1.3,1.1]])#weight1
         weights = tf.reduce_sum(class_weights*labels,axis=1)
         unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
         unweighted_losses_ss1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=ss1)
         unweighted_losses_ss2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=ss2)
         weighted_losses = unweighted_losses*weights + (unweighted_losses_ss1*weights + unweighted_losses_ss2*weights)*0.5
         regularization = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
         loss = tf.reduce_mean(weighted_losses,name = 'weighted_cross_entropy')
         loss = loss + regularization
         return loss 
     
def get_train_data(train_data_mat,batch_size,originT1Array,originIRArray,originFLAIRArray,originLabelArray,step):
    c=32
    data_t1 = np.zeros((batch_size,c,c,c,1)).astype(np.float32)
    data_t2 = np.zeros((batch_size,c,c,c,1)).astype(np.float32)
    data_t3 = np.zeros((batch_size,c,c,c,1)).astype(np.float32)
    label= np.zeros((batch_size,c,c,c,1)).astype(np.float32)   
    mat_len = list(range(0,len(train_data_mat)))
    del mat_len[len(mat_len)-c-batch_size+2:len(mat_len)]   
    for i in range(0,batch_size):
        x = np.random.randint(0,240-c+1)
        y = np.random.randint(0,240-c+1)
        z = np.random.randint(0, len(train_data_mat)-c+1)
        choice = np.random.randint(0,5)
        for j in range(0,c):
             data_mat_1 = originT1Array[train_data_mat[z+j]-1,x:x+c,y:y+c]
             data_mat_2 = originFLAIRArray[train_data_mat[z+j]-1,x:x+c,y:y+c]
             data_mat_3 = originIRArray[train_data_mat[z+j]-1,x:x+c,y:y+c]
             label_mat = originLabelArray[train_data_mat[z+j]-1,x:x+c,y:y+c]
             data_mat_1,data_mat_2,data_mat_3,label_mat = data_aug(data_mat_1,data_mat_2,data_mat_3,label_mat,choice)             
             data_t1[i,j,:,:,0] = data_mat_1
             data_t2[i,j,:,:,0] = data_mat_2
             data_t3[i,j,:,:,0] = data_mat_3
             label[i,j,:,:,0] = label_mat            
    return data_t1,data_t2,data_t3,label     

def get_val_data(val_mat,batch_size,originT1Array,originIRArray,originFLAIRArray,originLabelArray,x,y,z):
    c=32
    data_t1 = np.zeros((1,c,c,c,1)).astype(np.float32)
    data_t2 = np.zeros((1,c,c,c,1)).astype(np.float32)
    data_t3 = np.zeros((1,c,c,c,1)).astype(np.float32)
    label= np.zeros((1,c,c,c,1)).astype(np.float32)
    for i in range(0,1): 
        for j in range(c):   
             data_t1[i,j,:,:,0] = originT1Array[val_mat[z+j]-1,x:x+c,y:y+c]
             data_t2[i,j,:,:,0] = originFLAIRArray[val_mat[z+j]-1,x:x+c,y:y+c]
             data_t3[i,j,:,:,0] = originIRArray[val_mat[z+j]-1,x:x+c,y:y+c]
             label[i,j,:,:,0] = originLabelArray[val_mat[z+j]-1,x:x+c,y:y+c]       
    return data_t1,data_t2,data_t3,label

def get_test_data(val_data_mat,batchsize,originT1Array,originIRArray,originFLAIRArray,predictions):
    c=32
    data_t1 = np.zeros((batchsize,c,c,c,1)).astype(np.float32)
    data_t2 = np.zeros((batchsize,c,c,c,1)).astype(np.float32)
    data_t3 = np.zeros((batchsize,c,c,c,1)).astype(np.float32)  
    for i in range(batchsize):
        for j in range (c):
             x,y,z = predictions[i+1]
             data_t1[i,j,:,:,0] = originT1Array[val_data_mat[z+j]-1,x:x+c,y:y+c]
             data_t2[i,j,:,:,0] = originFLAIRArray[val_data_mat[z+j]-1,x:x+c,y:y+c]
             data_t3[i,j,:,:,0] = originIRArray[val_data_mat[z+j]-1,x:x+c,y:y+c]
    return data_t1,data_t2,data_t3

def data_aug(data_mat_1,data_mat_2,data_mat_3,label_mat,choice):
    if choice==0:
        data_mat_1 = data_mat_1
        data_mat_2 = data_mat_2
        data_mat_3 = data_mat_3
        label_mat = label_mat
    elif choice==1:
        data_mat_1 = np.fliplr(data_mat_1)
        data_mat_2 = np.fliplr(data_mat_2)
        data_mat_3 = np.fliplr(data_mat_3)
        label_mat = np.fliplr(label_mat)
    elif choice==2: 
        data_mat_1 = np.flipud(data_mat_1)
        data_mat_2 = np.flipud(data_mat_2)
        data_mat_3 = np.flipud(data_mat_3)
        label_mat = np.flipud(label_mat)
    elif choice==3:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation1(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==4:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation2(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==5:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation3(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==6:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation4(data_mat_1,data_mat_2,data_mat_3,label_mat)        
    return data_mat_1,data_mat_2,data_mat_3,label_mat

def data_augmentation1(image1,image2,image3,image4):
    [image1,image2,image3,image4] = np.expand_dims([image1,image2,image3,image4],-1)
    [image1,image2,image3,image4] = tl.prepro.rotation_multi([image1,image2,image3,image4] , rg=20, is_random=True, fill_mode='constant')        
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4

def data_augmentation3(image1,image2,image3,image4):
    [image1,image2,image3,image4] = np.expand_dims([image1,image2,image3,image4],-1)
    [image1,image2,image3,image4] = tl.prepro.elastic_transform_multi([image1,image2,image3,image4], alpha=720, sigma=24, is_random=True) 
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4

def data_augmentation2(image1,image2,image3,image4):
    [image1,image2,image3,image4] = np.expand_dims([image1,image2,image3,image4],-1)
    [image1,image2,image3,image4] = tl.prepro.shift_multi([image1,image2,image3,image4] ,  wrg=0.10,  hrg=0.10, is_random=True, fill_mode='constant')
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4 

def data_augmentation4(image1,image2,image3,image4):
    [image1,image2,image3,image4] = np.expand_dims([image1,image2,image3,image4],-1)   
    [image1,image2,image3,image4] = tl.prepro.zoom_multi([image1,image2,image3,image4] , zoom_range=[0.9, 1.1], is_random=True, fill_mode='constant')      
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32) 
    return image1,image2,image3,image4   
     