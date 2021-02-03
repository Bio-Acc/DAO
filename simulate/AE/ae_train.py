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
hidden_vecotr=model.encode()


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

    tf.global_variables_initializer().run()

    for epoch in range(200):
        print('epoch %d'%(epoch))
        batchfied_training_data=get_batch(x_train,256)
        training_loss_list=[]
        gc_loss_list=[]
        for i in batchfied_training_data:
            input_=i
            loss_,_=sess.run([decoder_loss,optimizer],feed_dict={X:input_})
            training_loss_list.append(loss_)
            
        print(np.mean(training_loss_list))
        
    '''
    l=len(x_train)//100
    result_h=[]
    for i in range(101):
        temp_input=x_train[l*i:min(l*(i+1),len(x_train))]
        #if not len(temp_input):
        #    break
        h_tr=sess.run([hidden_vector],feed_dict={X:temp_input})
        result_h.append(h_tr[0])
    h_tr=np.concatenate(result_h,axis=0)
    
    
    l=len(x_train)//100
    result_h=[]
    for i in range(101):
        temp_input=x_test[l*i:min(l*(i+1),len(x_test))]
        #if not len(temp_input):
        #    break
        h_te=sess.run([hidden_vector],feed_dict={X:temp_input})
        result_h.append(h_te[0])
    h_te=np.concatenate(result_h,axis=0)
    
    
    file=open('ae_embed_simulate_hidden.pkl','wb')
    pickle.dump(h_tr,file)
    pickle.dump(h_te,file)
    file.close()
    '''
    saver=tf.train.Saver()
    saver.save(sess,'save_model/ae_simulate.ckpt')

