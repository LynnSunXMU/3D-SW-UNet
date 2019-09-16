# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:25:48 2019

@author: mwa
"""

import tensorflow as tf
from layers import UNet_decoder_block_3t,UNet_encoder_block_3t,Conv3d,VFR_layer,MaxPooling3,UNet_encoder_ss1,UNet_encoder_ss2,Unet_decoder_deconv

def Unet_3d_SW(images_t1, images_t2,images_t3,batch_size , f_num , is_train = True, 
                    BN = False,reuse=False,size = 240,name = 'Unet',dropOut=1.0,is_test=False,is_ss = True,num_classes=4):

    with tf.variable_scope("Net_%s"%(name), reuse = reuse):
        #BN = False
        repeat_ =2
        sq_rate = 4
        h1_t1 = images_t1
        h1_t2 = images_t2  
        h1_t3 = images_t3
        h1_t1,h1_t2,h1_t3 = UNet_encoder_block_3t(h1_t1,h1_t2,h1_t3,0,f_num,repeat_,is_train,BN)        
        name_ = "Conv_ReLu_seg_encoder_t1_add1"
        h1_t1 =   tf.nn.relu(Conv3d(h1_t1, f_num*2, name = name_, is_train = is_train, BN = BN,ks=3))
        name_ = "Conv_ReLu_seg_encoder_t2_add2"
        h1_t2 =   tf.nn.relu(Conv3d(h1_t2, f_num*2, name = name_, is_train = is_train, BN = BN,ks=3)) 
        name_ = "Conv_ReLu_seg_encoder_t2_add3"
        h1_t3 =   tf.nn.relu(Conv3d(h1_t3, f_num*2, name = name_, is_train = is_train, BN = BN,ks=3))
        name_ = "VFR_layer_1"
        h1_t1,h1_t2,h1_t3 = VFR_layer(h1_t1,h1_t2,h1_t3,name_,is_train,BN,sq_rate)
               
        h2_level_t1 = h1_t1
        h2_t1 = MaxPooling3(h1_t1)
        h2_level_t2 = h1_t2
        h2_t2 = MaxPooling3(h1_t2)
        h2_level_t3 = h1_t3
        h2_t3 = MaxPooling3(h1_t3)
        
        if is_ss:
            h_ss1 = UNet_encoder_ss1(h2_t1,h2_t2,h2_t3,0,f_num,batch_size,size,is_train,BN,num_classes=num_classes)      
        h2_t1,h2_t2,h2_t3 = UNet_encoder_block_3t(h2_t1,h2_t2,h2_t3,1,f_num,repeat_,is_train,BN)
        name_ = "VFR_layer_2"
        h2_t1,h2_t2,h2_t3 = VFR_layer(h2_t1,h2_t2,h2_t3,name_,is_train,BN,sq_rate)

        h3_level_t1 = h2_t1
        h3_t1 = MaxPooling3(h2_t1)
        h3_level_t2 = h2_t2
        h3_t2 = MaxPooling3(h2_t2)
        h3_level_t3 = h2_t3
        h3_t3 = MaxPooling3(h2_t3)

        if is_ss:
            h_ss2 = UNet_encoder_ss2(h3_t1,h3_t2,h3_t3,1,f_num,batch_size,size,is_train,BN,num_classes=num_classes)
        h3_t1,h3_t2,h3_t3 = UNet_encoder_block_3t(h3_t1,h3_t2,h3_t3,2,f_num,repeat_,is_train,BN)
        name_ = "VFR_layer_3"
        h3_t1,h3_t2,h3_t3 = VFR_layer(h3_t1,h3_t2,h3_t3,name_,is_train,BN,sq_rate)
       
        h3_t1_,h3_t2_,h3_t3_ = Unet_decoder_deconv(h3_t1,h3_level_t1,h3_t2,h3_level_t2,h3_t3,h3_level_t3,size,f_num,1,batch_size)
        h3_t1_,h3_t2_,h3_t3_ = UNet_decoder_block_3t(h3_t1_,h3_t2_,h3_t3_,2,f_num,repeat_,is_train,BN) 
        name_ = "VFR_layer_4"
        h3_t1_,h3_t2_,h3_t3_ = VFR_layer(h3_t1_,h3_t2_,h3_t3_,name_,is_train,BN,sq_rate)        
        
        h2_t1_,h2_t2_,h2_t3_ = Unet_decoder_deconv(h3_t1_,h2_level_t1,h3_t2_,h2_level_t2,h3_t3_,h2_level_t3,size,f_num,0,batch_size)               
        h2_t1_,h2_t2_,h2_t3_ = UNet_decoder_block_3t(h2_t1_,h2_t2_,h2_t3_,1,f_num,repeat_,is_train,BN) 
        name_ = "VFR_layer_5"
        h2_t1_,h2_t2_,h2_t3_ = VFR_layer(h2_t1_,h2_t2_,h2_t3_,name_,is_train,BN,sq_rate)
                       
        h2_t1_ = tf.nn.dropout(h2_t1_,keep_prob=dropOut)
        h2_t2_ = tf.nn.dropout(h2_t2_,keep_prob=dropOut)
        h2_t3_ = tf.nn.dropout(h2_t3_,keep_prob=dropOut)

        h2_ = tf.concat([h2_t1_,h2_t2_,h2_t3_], 4)       
        name_ = "Conv_ReLu_%s_-0_seg_decoder_t2"%(0) 
        h2_ = tf.nn.relu(Conv3d(h2_, f_num, name = name_, is_train = is_train, BN = BN,ks=1))       
        h = Conv3d(h2_, num_classes, name = 'Conv3d_-1_seg_decoder',  is_train = is_train,BN=BN,ks=1)
        if is_test:
            return tf.nn.softmax(h)
        elif is_ss:
            return h,h_ss1,h_ss2
        else:
            return h