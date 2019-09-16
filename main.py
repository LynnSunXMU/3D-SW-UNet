#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 20:04:33 2018

@author: maw
"""


import os
import tensorflow as tf
import time
import numpy as np
from model import Unet_3d_SW
import ops
import SimpleITK as sitk

config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)

inputDir = "./data/MRBrainS13/"
labelDir = "./data/MRBrainS13/"
T1 = inputDir + "T1.nii"
T1_IR = inputDir + "T1_IR.nii"
T2_FLAIR = inputDir + "T2_FLAIR.nii"
label_path = labelDir + "label.nii"
originT1Image = sitk.ReadImage(T1)
originT1Array = sitk.GetArrayFromImage(originT1Image)
originIRImage = sitk.ReadImage(T1_IR)
originIRArray = sitk.GetArrayFromImage(originIRImage)
originFLAIRImage = sitk.ReadImage(T2_FLAIR)
originFLAIRArray = sitk.GetArrayFromImage(originFLAIRImage)
originLabelImage = sitk.ReadImage(label_path)
originLabelArray = sitk.GetArrayFromImage(originLabelImage).astype(np.float32)
    
def run(sess,Train):     
    if Train:
         batch_size=8
    else:
         batch_size=32
    max_step=17000
    val_subject = 5
    img_size = 32
    overlap = 16
    val_dic = {'1_start':1,'1_end':48,'2_start':49,'2_end':96,'3_start':97,
               '3_end':144,'4_start':145,'4_end':192,'5_start':193,'5_end':240}
    train_data_mat = list(range(1,241))
    val_subject_start = val_dic[str(val_subject) + '_start']
    val_subject_end = val_dic[str(val_subject) + '_end']
    del train_data_mat[val_subject_start-1:val_subject_end]
    val_data_mat = range(val_subject_start,val_subject_end+1)          
    with tf.name_scope('input'):
        x_t1 = tf.placeholder(tf.float32, [batch_size,img_size, img_size, img_size, 1], name='inputs') 
        x_t2 = tf.placeholder(tf.float32, [batch_size,img_size, img_size, img_size, 1], name='inputs') 
        x_t3 = tf.placeholder(tf.float32, [batch_size,img_size, img_size, img_size, 1], name='inputs') 
        y = tf.placeholder(tf.float32, [batch_size,img_size, img_size, img_size, 1], name='label')    
    if Train:
         logits,logits_ss1,logits_ss2=Unet_3d_SW(x_t1,x_t2,x_t3, batch_size, 32,  is_train = True,BN = True,reuse=False,size = img_size,name='Unet',dropOut=1.0,is_ss = True)    
         loss= ops.weighted_cross_entropy_ss(logits,logits_ss1,logits_ss2,y)
    else:
         logits=Unet_3d_SW(x_t1,x_t2,x_t3, batch_size, 32,  is_train = False,BN = True,reuse=False,size = img_size,name='Unet',is_test=False,is_ss = False) 
         loss = ops.weighted_cross_entropy(logits,y)
    loss_compute = ops.weighted_cross_entropy(logits,y)
    
    logs="./logs/"
    model_log="UNET_SEG_AUG/"
    logs_path=os.path.join(logs, model_log)
    writer = tf.summary.FileWriter(logs_path)
    writer.add_graph(sess.graph)
    global_step = tf.Variable(0, trainable = False)

    rate = tf.train.exponential_decay(learning_rate = 0.001,global_step = global_step,staircase = True,decay_steps = 15000,decay_rate = 0.1)
    var_list = tf.global_variables()
    print ("total vars is : %s"%(ops.cal_para(var_list)))
    train_opt = tf.train.AdamOptimizer(rate).minimize(loss,global_step = global_step)

    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    saver = tf.train.Saver()
    counter=0
    load_model = 1
    modelName = "MRBrainS13_SWN"
    checkpoint_dir="./train_model/" + modelName + "/"
    folder = os.path.exists(checkpoint_dir)
    if not folder:
        os.mkdir(checkpoint_dir) 
    checkpoint_dir=checkpoint_dir+str(val_subject) 
    folder = os.path.exists(checkpoint_dir)
    if not folder:
        os.mkdir(checkpoint_dir) 
    model_name="Unet_Seg.ckpt"
    if load_model == True:  
       ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
       if ckpt and ckpt.model_checkpoint_path:
           ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
           saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))    
           global_step = ckpt.model_checkpoint_path.split('-')[-1]
           print('Loading success, global_step is %s' % global_step)
           counter =  int(global_step)
    
    #for train
    if Train:
        result_file = os.path.join(checkpoint_dir,"result.txt")
        if not os.path.exists(result_file):
             file_w = open(result_file,'w')
        file_w = open(result_file,'r+')       
        file_w.read()
        file_w.write("total vars is : %s" % (ops.cal_para(var_list)) + '\n')
        file_w.close()        
       
        if counter>=0:
            for i in range(counter,max_step+1):
                a_batch_t1,a_batch_t2,a_batch_t3,b_batch=ops.get_train_data(train_data_mat,batch_size,originT1Array,originIRArray,originFLAIRArray,originLabelArray,i)
                if i%10 == 0:
                    train_loss=loss_compute.eval(session=sess,feed_dict={x_t1:a_batch_t1,x_t2:a_batch_t2,x_t3:a_batch_t3, y:b_batch})
                    print ("step %d, training loss %g" % (i, train_loss))
                sess.run(train_opt,feed_dict={x_t1:a_batch_t1,x_t2:a_batch_t2,x_t3:a_batch_t3, y:b_batch})
                if i%200==0:
                    model_path=os.path.join(checkpoint_dir, model_name)
                    saver_path=saver.save(sess,model_path,global_step=i)
                if i == max_step:
                    coord.request_stop()
                    coord.join(threads)
    else:
        time_start = time.time()
        overlap = 8
        saver=tf.train.Saver()
        model_name="Unet_Seg.ckpt-17000"
        model_path=os.path.join(checkpoint_dir, model_name)
        saver.restore(sess,model_path)
        predictions = {}
        full_prob = np.zeros((48,240,240,4))
        full_count = np.zeros((48,240,240,4))
        PathNum = 0        
        for i in range(1,batch_size+1):
            predictions[i] = []
        for l in range(0,240-img_size+1,overlap):
             for j in range(0,240-img_size+1,overlap):
                  for k in range(0,len(val_data_mat)-img_size+1,overlap):
                       PathNum += 1
                       predictions[PathNum] = (l,j,k)
                       if PathNum==batch_size:
                           a_batch_t1,a_batch_t2,a_batch_t3=ops.get_test_data(val_data_mat,batch_size,originT1Array,originIRArray,originFLAIRArray,predictions)
                           logit_1_np = sess.run(logits,feed_dict={x_t1:a_batch_t1,x_t2:a_batch_t2,x_t3:a_batch_t3}) 
                           PathNum = 0
                           for N in range(batch_size):
                               l1,j1,k1=predictions[N+1]
                               full_prob[k1:k1+img_size,l1:l1+img_size,j1:j1+img_size,:] += logit_1_np[N,:,:,:,:]
                               full_count[k1:k1+img_size,l1:l1+img_size,j1:j1+img_size,:] += 1
                       print(k,l,j)
        results = full_prob/full_count
        results = np.argmax(results, axis = 3)
        results = results.astype(float)        
        results[results==1] = 85
        results[results==2] = 170
        results[results==3] = 255
        results[results==85] = 2       
        results[results==170] = 3
        results[results==255] = 1
        niiPath = './data/result/' + modelName 
        folder = os.path.exists(niiPath)
        if not folder:
            os.mkdir(niiPath)
        niiPath = niiPath + '/' + str(val_subject)
        folder = os.path.exists(niiPath)
        if not folder:
            os.mkdir(niiPath) 
        niiPath = niiPath + '/{}.nii'.format(val_subject)
        resultImage = sitk.GetImageFromArray(results)
        sitk.WriteImage(resultImage,niiPath)
        time_end = time.time()
        print('totally cost:',time_end-time_start)
def main():
    #for test:set isTrain=0
    Train=1
    if Train:
        run(sess,Train)
    else:
        run(sess,Train)
  
if __name__ == '__main__':
    main()
    
