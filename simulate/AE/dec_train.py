import tensorflow as tf
from tensorflow.python.layers import base
import pickle
import pandas as pd
import numpy as np
from DAO_utils import ResBlock, Codebook, ResLayer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class AE(base.Layer):
    def __init__(self,X):
        super(AE,self).__init__()
        self._input=X
        self.encoder=ResBlock(600,[256,128,64,20],[(1,3),(2,4)])
        self.decoder=ResBlock(20,[64,128,256,600],[(1,3),(2,4)])

    def encode(self):
        if not '_encode' in self.__dict__:
            encoding=self.encoder(self._input)
            self._encode=encoding
            return self._encode
        else:
            return self._encode

    def decode(self):
        if not '_reconstruction' in self.__dict__:
            embed=self.encode()
            self._reconstruction=self.decoder(embed)
            return self._reconstruction
        else:
            return self._reconstruction
    
    def reconstruction_loss(self):
        if not '_recon_loss' in self.__dict__:
            X=self._input
            Y=self.decode()
            self._recon_loss=tf.reduce_mean(tf.norm(X-Y,axis=-1)**2)
            return self._recon_loss
        else:
            return self._recon_loss

    def optimizer(self):
        self._optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.reconstruction_loss(),var_list=tf.trainable_variables())
        return self._optimizer


X=tf.placeholder(tf.float32,shape=[None,600],name='input')
model=AE(X)
optimizer=model.optimizer()
decoder_loss=model.reconstruction_loss()
hidden_vector=model.encode()


centroids=tf.Variable(tf.zeros([20,20]),name='table')
ht=tf.expand_dims(hidden_vector,-2)
ct=tf.reshape(centroids,[1,20,20])
dist=tf.norm(ht-ct,axis=-1)
q = 1.0/(1.0+dist**2/1.0)**((1.0+1.0)/2.0)
q = tf.transpose(tf.transpose(q)/tf.reduce_sum(q,axis=1))
index=tf.argmax(q,axis=1)
p = tf.stop_gradient((q**2)/tf.reduce_sum(q,axis=0))
p = tf.transpose(tf.transpose(p)/tf.reduce_sum(p,axis=1))
assign_loss = tf.reduce_sum(p*tf.log(tf.div(p,q)),axis=-1)

loss=tf.reduce_mean(assign_loss)
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,var_list=tf.trainable_variables())


def get_batch(data,batch_size):
    result=[]
    col=list(range(data.shape[0]))
    np.random.shuffle(col)
    while len(col)>=batch_size:
        result.append(data[col[:batch_size]])
        col=col[batch_size:]
    return result

file=open('../simulate_data.pkl','rb')
x_train=pickle.load(file)
y_train=pickle.load(file)
x_test=pickle.load(file)
y_test=pickle.load(file)
file.close()



config=tf.ConfigProto()
#config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    ae_vars=ae_vars[:-1]

    file=open('simulate_AE-kmeans_result.pkl','rb')
    cb=pickle.load(file)
    file.close()

    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(centroids,cb))

    saver=tf.train.Saver(ae_vars)
    saver.restore(sess,'save_model/ae_simulate.ckpt')
    
    for epoch in range(500):
        print('epoch %d'%(epoch))
        batchfied_training_data=get_batch(x_train,32)
        training_loss_list=[]
        gc_loss_list=[]
        for i in batchfied_training_data:
            input_=i
            loss_,_=sess.run([loss,optimizer],feed_dict={X:input_})
            training_loss_list.append(loss_)
  
        print(np.mean(training_loss_list))
        
    '''
    l=len(x_train)//100
    result_h=[]
    for i in range(101):
        temp_input=x_train[l*i:min(l*(i+1),len(x_train))]
        #if not len(temp_input):
        #    break
        h_tr=sess.run([index],feed_dict={X:temp_input})
        result_h.append(h_tr[0])
    h_tr=np.concatenate(result_h,axis=0)
    
    
    l=len(x_test)//100
    result_h=[]
    for i in range(101):
        temp_input=x_test[l*i:min(l*(i+1),len(x_test))]
        #if not len(temp_input):
        #    break
        h_te=sess.run([index],feed_dict={X:temp_input})
        result_h.append(h_te[0])
    h_te=np.concatenate(result_h,axis=0)
    
    file=open('dec_embed_simulate_index_20.pkl','wb')
    pickle.dump(h_tr,file)
    pickle.dump(h_te,file)
    file.close()
    '''
    saver=tf.train.Saver()
    saver.save(sess,'save_model/dec_simulate.ckpt')

