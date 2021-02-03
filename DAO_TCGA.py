import tensorflow as tf
from tensorflow.python.layers import base
import pickle
import pandas as pd
import numpy as np
from DAO_utils import ResBlock, Codebook, ResLayer


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import sys
CN=int(sys.argv[1])

class DAO(base.Layer):
    def __init__(self,X):
        super(DAO,self).__init__()
        self._input=X
        self.encoder=ResBlock(81574,[1000,500,100,100],[(1,3),(2,4)])
        self.decoder=ResBlock(100,[100,200,400,800,81574],[(1,3),(2,4)])
        self.codebook=Codebook(100,CN)
    
    def encode(self):
        if not '_encode' in self.__dict__:
            encoding=self.encoder(self._input)
            self._encode=encoding
            return self._encode
        else:
            return self._encode

    def vector_quantization(self):
        if not '_VQ' in self.__dict__:
            self._VQ=self.codebook(self.encode())
            return self._VQ
        else:
            return self._VQ
    
    def decode(self):
        if not '_reconstruction' in self.__dict__:
            VQ=self.vector_quantization()
            embed=self.encode()
            discrete_embed=VQ[0]
            dynamic_distance=tf.expand_dims(1./tf.stop_gradient(tf.norm(embed-discrete_embed,axis=-1)**2),axis=-1)
            gradient_copy_embed=dynamic_distance*embed+tf.stop_gradient(discrete_embed)-dynamic_distance*tf.stop_gradient(embed)
            self._reconstruction=self.decoder(gradient_copy_embed)
            return self._reconstruction
        else:
            return self._reconstruction

    def reconstruction_loss(self):
        if not '_recon_loss' in self.__dict__:
            X=self._input
            Y=self.decode()
            self._recon_loss=tf.reduce_mean(tf.norm((X-Y)/tf.clip_by_value(X,1,1000),axis=-1)**2)
            return self._recon_loss
        else:
            return self._recon_loss

    def VQ_loss(self):
        if not '_vq_loss' in self.__dict__:
            VQ=self.vector_quantization()
            embed=self.encode()
            discrete_embed=VQ[0]
            self._vq_loss=tf.reduce_mean(tf.norm(tf.stop_gradient(embed)-discrete_embed,axis=-1)**2)
            return self._vq_loss
        else:
            return self._vq_loss
    
    def commit_loss(self):
        if not '_commit_loss' in self.__dict__:
            VQ=self.vector_quantization()
            embed=self.encode()
            discrete_embed=VQ[0]
            self._commit_loss=tf.reduce_mean(tf.norm(embed-tf.stop_gradient(discrete_embed),axis=-1)**2)
            return self._commit_loss
        else:
            return self._commit_loss

    def codebook_loss(self):
        if not '_cb_loss' in self.__dict__:
            cb=self.codebook.lookup_table
            embed=self.encode()
            self._cb_loss=tf.reduce_mean(tf.norm(cb-tf.reduce_mean(tf.stop_gradient(embed),axis=0),axis=-1)**2)
            return self._cb_loss
        else:
            return self._cb_loss

    def Loss(self):
        if not '_loss' in self.__dict__:
            self._loss=self.reconstruction_loss()+1.0*self.VQ_loss()+1.0*self.commit_loss()+0.001*self.codebook_loss()
            return self._loss
        else:
            return self._loss

    def optimizer(self):
        self._optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.Loss(),var_list=tf.trainable_variables())
        return self._optimizer



X=tf.placeholder(tf.float32,shape=[None,81574],name='input')
model=DAO(X)
optimizer=model.optimizer()
index=model.vector_quantization()[1]
decoder_loss=model.reconstruction_loss()
hidden_vector=model.encode()
codebook=model.codebook.lookup_table


def get_batch(data,batch_size):
    result=[]
    col=list(data.columns)
    np.random.shuffle(col)
    while len(col)>=batch_size:
        result.append(data.loc[:,col[:batch_size]].values)
        col=col[batch_size:]
    return result

file=open('../TCGA/TCGA_ex_sp100all_mean_normalis0.pkl','rb')
data=pickle.load(file)
file.close()
#data=data.iloc[:,7:]
data=data.iloc[:,7:]





config=tf.ConfigProto()
#config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    
    
    tf.global_variables_initializer().run()
    
    for epoch in range(200):
        print('epoch %d'%(epoch))
        batchfied_training_data=get_batch(data,32)
        training_loss_list=[]
        for i in batchfied_training_data:
            input_=i[:-3].transpose()
            loss_,_,index_=sess.run([decoder_loss,optimizer,index],feed_dict={X:input_})
            training_loss_list.append(loss_)
        print(np.mean(training_loss_list))
        print(index_,flush=True)
    '''
    index_,h_=sess.run([index,hidden_vector],feed_dict={X:data[:-3].transpose})
    file=open('DAO_TCGA_{}.pkl'.format(CN),'wb')
    pickle.dump(index_,file)
    file.close()
    file=open('DAO_TCGA_hidden.pkl'.format(CN),'wb')
    pickle.dump(h_,file)
    file.close()
    cb=sess.run(codingtable.lookup_table)
    file=open('DAO_TCGA_codebook.pkl'.format(CN),'wb')
    pickle.dump(cb,file)
    file.close()
    '''
    
    saver=tf.train.Saver()
    saver.save(sess,'save_model/DAO_TCGA_{}.ckpt'.format(CN))
    
