# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 11:23:36 2019

@author: mwa
"""

import tensorflow as tf

def MaxPooling3(value, ksize = [1, 3, 3, 3, 1], strides = [1, 2, 2, 2, 1],
               padding = 'SAME', name = 'MaxPooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool3d(value, ksize = ksize,
                              strides = strides, padding = padding)

def Conv3d(x,num_filter,name = ' ', is_train = True, BN=True,ks=3,stride=1):
     with tf.variable_scope(name):
        fil_num=num_filter
        shape_=x.get_shape().as_list()
        dim=shape_[-1]
        w = create_variables(name=name + 'conv', shape=[ks,ks,ks,dim,fil_num])
        b = create_bias(      name= name+ 'bias', shape = [fil_num])   
        name_ =  '_Conv_First'
        result1 = Conv3d_nop(x, w, b,  name = name+ name_,stride=stride) 
        if BN:
            result1 = BatchNorm(result1, is_train, name = name + name_ ,epsilon = 1e-5, momentum = 0.9)
        else:
            result1 = result1
     return result1

def Deconv3d(value, output_shape, k_d = 3,k_h = 3, k_w = 3,  strides =[1, 2, 2, 2, 1],
             name = 'Deconv3d', with_w = False):
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, k_d, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv3d_transpose(value, weights,
                                        output_shape, strides = strides)
        biases = bias(name+'biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv
        
def BatchNorm(value, is_train = True, name = 'BatchNorm',
              epsilon = 1e-5, momentum = 0.9):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name+'bn'
        )
        
def Conv3d_nop(input_,w, b,ks=3,stride = 1,name="conv3d_op", reuse = False):
   conv = tf.nn.conv3d(input_, w, strides=[1, stride,  stride, stride, 1], padding='SAME', name=name + "_conv3d_op")
   return tf.add(conv, b, name = name+"_add_op")

def create_bias(name, shape):
    new_variables = tf.get_variable(initializer= 0.0,   name = name+"bias" )      
    return  new_variables

def weight(name, shape, stddev = 0.02, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
         regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
    return var

def bias(name, shape, bias_start = 0.1, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32))
    return var

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    new_variables = tf.get_variable(name=name+"conv", shape=shape, initializer=initializer,
                                    regularizer=None)
    return new_variables

def VFR_layer(input_tensor1,input_tensor2,input_tensor3,name,is_train,BN,sq_rate = 4):

    def f(old,input):
        h1 = input[0]
        h2 = input[1]
        h3 = input[2]
        kernel_sample = input[3]
        kernel_sample2 = input[4]
        
        h_axis1 = tf.concat([h1,h2,h3],3)
        conv_axis1 = tf.reduce_mean(h_axis1,axis = [1,2],keep_dims = True)
        conv1_axis1 = tf.nn.relu(tf.nn.conv2d(conv_axis1,kernel_sample[:,:,0,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis1 = tf.nn.relu(tf.nn.conv2d(conv_axis1,kernel_sample[:,:,1,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis1 = tf.nn.relu(tf.nn.conv2d(conv_axis1,kernel_sample[:,:,2,:,:],[1,1,1,1],padding = "SAME"))
        conv1_axis1 = tf.nn.sigmoid(tf.nn.conv2d(conv1_axis1,kernel_sample2[:,:,0,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis1 = tf.nn.sigmoid(tf.nn.conv2d(conv2_axis1,kernel_sample2[:,:,1,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis1 = tf.nn.sigmoid(tf.nn.conv2d(conv3_axis1,kernel_sample2[:,:,2,:,:],[1,1,1,1],padding = "SAME"))
     
        h_axis2 = tf.concat([h1,h2,h3],2)
        conv_axis2 = tf.reduce_mean(h_axis2,axis = [1,3],keep_dims = True)
        conv_axis2 = tf.transpose(conv_axis2,[0,1,3,2])
        conv1_axis2 = tf.nn.relu(tf.nn.conv2d(conv_axis2,kernel_sample[:,:,3,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis2 = tf.nn.relu(tf.nn.conv2d(conv_axis2,kernel_sample[:,:,4,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis2 = tf.nn.relu(tf.nn.conv2d(conv_axis2,kernel_sample[:,:,5,:,:],[1,1,1,1],padding = "SAME"))
        conv1_axis2 = tf.nn.sigmoid(tf.nn.conv2d(conv1_axis2,kernel_sample2[:,:,3,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis2 = tf.nn.sigmoid(tf.nn.conv2d(conv2_axis2,kernel_sample2[:,:,4,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis2 = tf.nn.sigmoid(tf.nn.conv2d(conv3_axis2,kernel_sample2[:,:,5,:,:],[1,1,1,1],padding = "SAME"))
        
        h_axis3 = tf.concat([h1,h2,h3],1)
        conv_axis3 = tf.reduce_mean(h_axis3,axis = [2,3],keep_dims = True)
        conv_axis3 = tf.transpose(conv_axis3,[0,2,3,1])
        conv1_axis3 = tf.nn.relu(tf.nn.conv2d(conv_axis3,kernel_sample[:,:,6,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis3 = tf.nn.relu(tf.nn.conv2d(conv_axis3,kernel_sample[:,:,7,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis3 = tf.nn.relu(tf.nn.conv2d(conv_axis3,kernel_sample[:,:,8,:,:],[1,1,1,1],padding = "SAME"))
        conv1_axis3 = tf.nn.sigmoid(tf.nn.conv2d(conv1_axis3,kernel_sample2[:,:,6,:,:],[1,1,1,1],padding = "SAME"))
        conv2_axis3 = tf.nn.sigmoid(tf.nn.conv2d(conv2_axis3,kernel_sample2[:,:,7,:,:],[1,1,1,1],padding = "SAME"))
        conv3_axis3 = tf.nn.sigmoid(tf.nn.conv2d(conv3_axis3,kernel_sample2[:,:,8,:,:],[1,1,1,1],padding = "SAME"))
          
        conv1_axis2 = tf.transpose(conv1_axis2,[0,1,3,2])
        conv2_axis2 = tf.transpose(conv2_axis2,[0,1,3,2])
        conv3_axis2 = tf.transpose(conv3_axis2,[0,1,3,2])
     
        conv1_axis3 = tf.transpose(conv1_axis3,[0,3,1,2])
        conv2_axis3 = tf.transpose(conv2_axis3,[0,3,1,2])
        conv3_axis3 = tf.transpose(conv3_axis3,[0,3,1,2])

        mat1 = tf.multiply(conv1_axis1,h1)
        mat1 = tf.multiply(conv1_axis2,mat1)
        mat1 = tf.multiply(conv1_axis3,mat1)
        mat1 = tf.expand_dims(mat1,-1)
         
        mat2 = tf.multiply(conv2_axis1,h2)
        mat2 = tf.multiply(conv2_axis2,mat2)
        mat2 = tf.multiply(conv2_axis3,mat2)
        mat2 = tf.expand_dims(mat2,-1)
         
        mat3 = tf.multiply(conv3_axis1,h3)
        mat3 = tf.multiply(conv3_axis2,mat3)
        mat3 = tf.multiply(conv3_axis3,mat3)
        mat3 = tf.expand_dims(mat3,-1)
        
        mat = tf.concat([mat1,mat2,mat3],4)
        
        return mat
   
    batch_size = int(input_tensor1.shape[0])
    size = int(input_tensor1.shape[1])
    dims = int(input_tensor1.shape[4])
    kernel1 = create_variables(name=name + 'conv_VFR1', shape=[1,1,dims,9,size*3,size*3/4])
    kernel2 = create_variables(name=name + 'conv_VFR2', shape=[1,1,dims,9,size*3/4,size])

    x1 = input_tensor1
    x2 = input_tensor2
    x3 = input_tensor3

    kernel_scan1 = tf.transpose(kernel1,[2,0,1,3,4,5])
    kernel_scan2 = tf.transpose(kernel2,[2,0,1,3,4,5])

    x_scan1 = tf.transpose(x1,[4,0,1,2,3])
    x_scan2 = tf.transpose(x2,[4,0,1,2,3])
    x_scan3 = tf.transpose(x3,[4,0,1,2,3])

    mat = tf.scan(f, (x_scan1,x_scan2,x_scan3,kernel_scan1,kernel_scan2) ,initializer = tf.zeros((batch_size,size,size,size,3)))     
    c1 = mat[:,:,:,:,:,0]
    c2 = mat[:,:,:,:,:,1]
    c3 = mat[:,:,:,:,:,2]
    
    c1 = tf.transpose(c1,[1,2,3,4,0])
    c2 = tf.transpose(c2,[1,2,3,4,0])
    c3 = tf.transpose(c3,[1,2,3,4,0])
       
    return c1,c2,c3

def UNet_encoder_block_3t(h1_t1,h1_t2,h1_t3,l,f_num,repeat_,is_train,BN):
    f_num = f_num * 2 ** l
    for i in range(repeat_):
        name_ = "Conv_ReLu_%s_seg_encoder_t1_%s"%(i,l) 
        h1_t1 =   tf.nn.relu(Conv3d(h1_t1, f_num+f_num*i, name = name_, is_train = is_train, BN = BN,ks=3))
        name_ = "Conv_ReLu_%s_seg_encoder_t2_%s"%(i,l) 
        h1_t2 =   tf.nn.relu(Conv3d(h1_t2, f_num+f_num*i, name = name_, is_train = is_train, BN = BN,ks=3))  
        name_ = "Conv_ReLu_%s_seg_encoder_t3_%s"%(i,l) 
        h1_t3 =   tf.nn.relu(Conv3d(h1_t3, f_num+f_num*i, name = name_, is_train = is_train, BN = BN,ks=3))             
    return h1_t1,h1_t2,h1_t3

def UNet_decoder_block_3t(h3_t1_,h3_t2_,h3_t3_,l,f_num,repeat_,is_train,BN):
    f_num = f_num * 2 ** l
    for i in range(repeat_):
        name_ = "Conv_ReLu_%s_seg_decoder_t1_%s"%(i,l)#30
        h3_t1_ =   tf.nn.relu(Conv3d(h3_t1_,  f_num, name = name_, is_train = is_train, BN = BN,ks=3))           
        name_ = "Conv_ReLu_%s_seg_decoder_t2_%s"%(i,l)#30
        h3_t2_ =   tf.nn.relu(Conv3d(h3_t2_,  f_num, name = name_, is_train = is_train, BN = BN,ks=3))
        name_ = "Conv_ReLu_%s_seg_decoder_t3_%s"%(i,l)#30
        h3_t3_ =   tf.nn.relu(Conv3d(h3_t3_,  f_num, name = name_, is_train = is_train, BN = BN,ks=3))               
    return h3_t1_,h3_t2_,h3_t3_

def UNet_encoder_ss1(h2_t1,h2_t2,h2_t3,l,f_num,batch_size,size,is_train,BN,num_classes=4):
    name_ = "Conv_ReLu_seg_decoder_t1_ss_%s"%(l)
    h2_t1_ss = Deconv3d(h2_t1, output_shape = [batch_size, size, size, size, f_num*2], name = name_)   
    name_ = "Conv_ReLu_seg_decoder_t2_ss_%s"%(l)
    h2_t2_ss = Deconv3d(h2_t2, output_shape = [batch_size, size, size, size, f_num*2], name = name_) 
    name_ = "Conv_ReLu_seg_decoder_t3_ss_%s"%(l)
    h2_t3_ss = Deconv3d(h2_t3, output_shape = [batch_size, size, size, size, f_num*2], name = name_)    
    ss_conv1 = tf.concat([h2_t1_ss,h2_t2_ss,h2_t3_ss],4)    
    name_ = "Conv_ss_%s"%(l)
    ss_conv1 = tf.nn.relu(Conv3d(ss_conv1, f_num, name = name_, is_train = is_train, BN = BN,ks=1))        
    h_ss1 = Conv3d(ss_conv1,  num_classes, name = 'Conv_ss_result_%s'%(l),  is_train = is_train,BN=BN,ks=1)    
    return h_ss1

def UNet_encoder_ss2(h3_t1,h3_t2,h3_t3,l,f_num,batch_size,size,is_train,BN,num_classes=4):
    name_ = "Conv_ReLu_seg_decoder_t1_ss_%s"%(l)
    h3_t1_ss = Deconv3d(h3_t1, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t1_ss_%s"%(l)
    h3_t1_ss = Deconv3d(h3_t1_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_seg_decoder_t2_ss_%s"%(l)
    h3_t2_ss = Deconv3d(h3_t2, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t2_ss_%s"%(l)
    h3_t2_ss = Deconv3d(h3_t2_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_) 
    name_ = "Conv_ReLu_seg_decoder_t3_ss_%s"%(l)
    h3_t3_ss = Deconv3d(h3_t3, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t3_ss_%s"%(l)
    h3_t3_ss = Deconv3d(h3_t3_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_)      
    ss_conv2 = tf.concat([h3_t1_ss,h3_t2_ss,h3_t3_ss],4)    
    name_ = "Conv_ss_%s"%(l)
    ss_conv2 = tf.nn.relu(Conv3d(ss_conv2, f_num, name = name_, is_train = is_train, BN = BN,ks=1))        
    h_ss2 = Conv3d(ss_conv2,  num_classes, name = 'Conv_ss_result_%s'%(l),  is_train = is_train,BN=BN,ks=1)    
    return h_ss2

def UNet_encoder_ss3(h4_t1,h4_t2,h4_t3,l,f_num,batch_size,size,is_train,BN,num_classes=4):
    
    name_ = "Conv_ReLu_2_seg_decoder_t1_ss3_%s"%(l)
    h3_t1_ss = Deconv3d(h4_t1, output_shape = [batch_size, size//4, size//4, size//4, f_num*2*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t1_ss3_%s"%(l)
    h3_t1_ss = Deconv3d(h3_t1_ss, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_0_seg_decoder_t1_ss3_%s"%(l)
    h3_t1_ss = Deconv3d(h3_t1_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_)
    
    name_ = "Conv_ReLu_2_seg_decoder_t2_ss3_%s"%(l)    
    h3_t2_ss = Deconv3d(h4_t2, output_shape = [batch_size, size//4, size//4, size//4, f_num*2*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t2_ss3_%s"%(l)    
    h3_t2_ss = Deconv3d(h3_t2_ss, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_0_seg_decoder_t2_ss3_%s"%(l)
    h3_t2_ss = Deconv3d(h3_t2_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_)  
    
    name_ = "Conv_ReLu_2_seg_decoder_t3_ss3_%s"%(l)    
    h3_t3_ss = Deconv3d(h4_t3, output_shape = [batch_size, size//4, size//4, size//4, f_num*2*2*2], name = name_)
    name_ = "Conv_ReLu_1_seg_decoder_t3_ss3_%s"%(l)    
    h3_t3_ss = Deconv3d(h3_t3_ss, output_shape = [batch_size, size//2, size//2, size//2, f_num*2*2], name = name_)
    name_ = "Conv_ReLu_0_seg_decoder_t3_ss3_%s"%(l)
    h3_t3_ss = Deconv3d(h3_t3_ss, output_shape = [batch_size, size, size, size, f_num*2*2], name = name_) 
     
    ss_conv2 = tf.concat([h3_t1_ss,h3_t2_ss,h3_t3_ss],4)    
    name_ = "Conv_ss_%s"%(l)
    ss_conv2 = tf.nn.relu(Conv3d(ss_conv2, f_num, name = name_, is_train = is_train, BN = BN,ks=1))        
    h_ss3 = Conv3d(ss_conv2,  num_classes, name = 'Conv_ss_result_%s'%(l),  is_train = is_train,BN=BN,ks=1)    
    return h_ss3

def Unet_decoder_deconv(h3_t1,h3_level_t1,h3_t2,h3_level_t2,h3_t3,h3_level_t3,size,f_num,l,batch_size):
    mul = 2 ** l
    name_ = "DeConv_ReLu_seg_decoder_t1_%s"%(l)
    h3_t1_ = Deconv3d(h3_t1, output_shape = [batch_size, size//mul,size//mul, size//mul, f_num*mul], name = name_)#120
    h3_t1_ = tf.concat([h3_level_t1,h3_t1_], 4)
    name_ = "DeConv_ReLu_seg_decoder_t2_%s"%(l)
    h3_t2_ = Deconv3d(h3_t2, output_shape = [batch_size, size//mul,size//mul, size//mul, f_num*mul], name = name_)#120
    h3_t2_ = tf.concat([h3_level_t2,h3_t2_], 4)
    name_ = "DeConv_ReLu_seg_decoder_t3_%s"%(l)
    h3_t3_ = Deconv3d(h3_t3, output_shape = [batch_size, size//mul,size//mul, size//mul, f_num*mul], name = name_)#120
    h3_t3_ = tf.concat([h3_level_t3,h3_t3_], 4)
    return h3_t1_,h3_t2_,h3_t3_

