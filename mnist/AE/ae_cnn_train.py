import tensorflow as tf
from tensorflow.python.layers import base
import pickle
import pandas as pd
import numpy as np
from DAO_utils import cnn_encoder, Codebook, ResLayer
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class AE(base.Layer):
    def __init__(self,X):
        super(AE,self).__init__()
        self._input=X
        self.encoder=cnn_encoder(strides=[1,2,2,1,1],filter_size=[5,3,3,5,3],channels=[16,32,64,64,64])
        self.decoder=cnn_encoder(encoder=False,strides=[1,2,2,1],filter_size=[3,3,3,3],channels=[16,16,32,64])
        self.FC1=ResLayer(input_size=7*7*64,res_size=[],output_size=50)
        self.FC2=ResLayer(input_size=50,res_size=[],output_size=7*7*64)

    def encode(self):
        if not '_encode' in self.__dict__:
            encoding=self.encoder(self._input)
            encoding=tf.reshape(encoding,[-1,7*7*64])
            encoding=self.FC1(encoding,[])
            self._encode=encoding
            return self._encode
        else:
            return self._encode

    def decode(self):
        if not '_reconstruction' in self.__dict__:
            embed=self.encode()
            decoding=self.FC2(embed,[])
            decoding=tf.reshape(decoding,[-1,7,7,64])
            self._reconstruction=self.decoder(decoding)
            return self._reconstruction
        else:
            return self._reconstruction
    
    def reconstruction_loss(self):
        if not '_recon_loss' in self.__dict__:
            X=tf.reshape(self._input,[-1,28*28])
            Y=tf.reshape(self.decode(),[-1,28*28])
            self._recon_loss=tf.reduce_mean(tf.norm(X-Y,axis=-1)**2)
            return self._recon_loss
        else:
            return self._recon_loss

    def optimizer(self):
        self._optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.reconstruction_loss(),var_list=tf.trainable_variables())
        return self._optimizer


X=tf.placeholder(tf.float32,shape=[None,28,28,1],name='input')
model=AE(X)
optimizer=model.optimizer()
decoder_loss=model.reconstruction_loss()
hidden_vector=model.encode()


def get_batch(data,batch_size):
    result=[]
    col=list(range(data.shape[0]))
    np.random.shuffle(col)
    while len(col)>=batch_size:
        result.append(data[col[:batch_size]])
        col=col[batch_size:]
    return result

file=open('../stimulate_data.pkl','rb')
x_train=pickle.load(file)
y_train=pickle.load(file)
x_test=pickle.load(file)
y_test=pickle.load(file)
file.close()


config=tf.ConfigProto()
#config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    tf.global_variables_initializer().run()

    for epoch in range(1000):
        print('epoch %d'%(epoch))
        batchfied_training_data=get_batch(x_train,8)
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
    
    
    file=open('ae_embed_mnist_hidden.pkl','wb')
    pickle.dump(h_tr,file)
    pickle.dump(h_te,file)
    file.close()
    '''
    saver=tf.train.Saver()
    saver.save(sess,'save_model/ae_simulate.ckpt')

