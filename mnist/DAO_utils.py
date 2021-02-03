import tensorflow as tf
from tensorflow.python.layers import base
import pickle
import pandas as pd
import numpy as np

class Layer_norm(base.Layer):
    def __init__(self):
        super(Layer_norm,self).__init__()
        self.gamma=tf.Variable(tf.ones(1))
        self.beta=tf.Variable(tf.zeros(1))
    def __call__(self,x):
        m,d=tf.nn.moments(x,axes=[-1])
        x_t=tf.transpose(x)
        x_t=(self.gamma/d)*(x_t-m)+self.beta
        x=tf.transpose(x_t)
        #x=self.gamma*(x/d)+self.beta
        return x

class ResLayer(base.Layer):
    def __init__(self,input_size,res_size,output_size,activation=tf.contrib.keras.layers.LeakyReLU(),dropout_rate=None,layer_norm=True,training=False,name=None):
        super(ResLayer,self).__init__(name=name)
        self.dropout_rate=dropout_rate
        self.training=training
        self.activation=activation
        self.norm=layer_norm
        self.params=[]
        #self.trainable_variables=[]
        if self.norm:
            self.layer_norm=Layer_norm()
        self.params.append(self.layer_norm.gamma)
        self.params.append(self.layer_norm.beta)
        self.w=tf.Variable(tf.truncated_normal([input_size,output_size]),name='weight')
        self.b=tf.Variable(tf.zeros([output_size]),name='bias')
        self.params.append(self.w)
        self.params.append(self.b)
        #self.trainable_variables.append(self.w)
        #self.trainable_variables.append(self.b)
        self.res_trans=[None]*len(res_size)
        for index,res_size_ in enumerate(res_size):
            if res_size_!=output_size:
                self.res_trans[index]=tf.Variable(tf.truncated_normal([res_size_,output_size]),name='res_weight')
                #self.trainable_variables.append(self.res_trans[index])
                self.params.append(self.res_trans[index])
    def __call__(self,input_x,res_x):
        assert len(res_x)==len(self.res_trans),'input residual number {} is not the same as structure {}'.format(len(res_x),len(self.res_trans))
        y=tf.matmul(input_x,self.w)+self.b
        index=0
        for i in self.res_trans:
            if i:
                y+=tf.matmul(res_x[index],i)
                index+=1
            else:
                y+=res_x[index]
                index+=1
        #if self.dropout_rate:
        #    y=tf.cond(self.training,lambda: tf.nn.dropout(y,keep_prob=self.dropout_rate),lambda: y)
        if self.norm:
            y=self.layer_norm(y)
        if self.activation:
            y=self.activation(y)
        return y

class ResBlock(base.Layer):
    def __init__(self,input_size,hidden_sizes,res_connection,dropout_rate=0.5,layer_norm=True,training=False,name=None):
        super(ResBlock,self).__init__(name=name)
        self.layers=[]
        self.res_table=[]
        #self.trainable_variables=[]
        self.params=[]
        for i in range(len(hidden_sizes)):
            self.res_table.append([])
        self.res_sort(hidden_sizes,res_connection)
        #print(self.res_table)
        sizes=[input_size]
        sizes.extend(hidden_sizes)
        for i in range(len(self.res_table)):
            temp_size=[]
            for j in self.res_table[i]:
                temp_size.append(sizes[j])
            self.layers.append(ResLayer(input_size,temp_size,hidden_sizes[i],dropout_rate=dropout_rate,layer_norm=layer_norm,training=training,name=name))
            #print('layer: {} {}'.format(input_size,temp_size))
            input_size=hidden_sizes[i]
        #self.trainable_variables.append(self.layers[-1].trainable_variables)
        self.params.extend(self.layers[-1].params)
    def __call__(self,x):
        result=[x]
        for i in range(len(self.layers)):
            temp_index=self.res_table[i]
            temp_res_input=[]
            for j in temp_index:
                temp_res_input.append(result[j])
            result.append(self.layers[i](result[-1],temp_res_input))
        return result[-1]
    def res_sort(self,hidden_sizes,res_connection):
        for i in res_connection:
            assert i[1]>i[0],'invalid residual direction'
            self.res_table[i[1]-1].append(i[0])
            for i in self.res_table:
                i.sort()
    def momentum_update(self,encoder,step):
        op=[]
        for i,j in zip(self.params,encoder.params):
            op.append(tf.assign(i,i*(1-step)+step*j))
        return op
    def initiate(self,encoder):
        op=[]
        for i,j in zip(self.params,encoder.params):
            op.append(tf.assign(i,j))
        return op

class cnn_encoder(base.Layer):
    def __init__(self,encoder=True,input_shape=[-1,28,28,1],strides=[1,2,2],filter_size=[3,3,3],channels=[16,32,64]):
        self.encoder=encoder
        self.activation=tf.contrib.keras.layers.LeakyReLU()
        self.filters=[]
        self.biass=[]
        self.norm_layer=[]
        self.strides=strides
        self.shapes=[input_shape]
        for i in range(len(self.strides)):
            shape=[-1,self.shapes[-1][1]//strides[i],self.shapes[-1][2]//strides[i],channels[i]]
            filter=tf.Variable(tf.truncated_normal([filter_size[i],filter_size[i],self.shapes[-1][-1],channels[i]],mean=0,stddev=0.1),name='filter')
            self.filters.append(filter)
            self.norm_layer.append(tf.layers.BatchNormalization())
            self.shapes.append(shape)
        if self.encoder:
            self.shapes=self.shapes[1:]
        else:
            self.shapes=self.shapes[::-1]
            self.shapes=self.shapes[1:]
            self.filters=self.filters[::-1]
            self.strides=self.strides[::-1]

        for i in self.shapes:
            bias=tf.Variable(tf.zeros(i[1:]),name='bias')
            self.biass.append(bias)
    
    def __call__(self,x):
        if self.encoder:
            for i in range(len(self.filters)):
                x=tf.nn.conv2d(x,filter=self.filters[i],strides=[1,self.strides[i],self.strides[i],1],data_format='NHWC',padding='SAME')
                x+=self.biass[i]
                x=self.norm_layer[i](x)
                x=self.activation(x)
        else:
            batch_size=tf.shape(x)[0]
            for i in range(len(self.filters)):
                x=tf.nn.conv2d_transpose(x,filter=self.filters[i],output_shape=[batch_size]+self.shapes[i][1:],data_format='NHWC',strides=[1,self.strides[i],self.strides[i],1],padding='SAME')
                x+=self.biass[i]
                x=self.norm_layer[i](x)
                x=self.activation(x)
        return x






class Codebook(base.Layer):
    def __init__(self,hidden_size,classes,name=None):
        super(Codebook,self).__init__(name=name)
        self.hidden_size=hidden_size
        self.classes=classes
        self.lookup_table=tf.Variable(tf.truncated_normal([classes,hidden_size],mean=0,stddev=0.1),name='table')
        self.trainable_variables.append(self.lookup_table)

    def __call__(self,x):
        t=tf.expand_dims(x,-2)
        temp_table=tf.reshape(self.lookup_table,[1,self.classes,self.hidden_size])
        dist=tf.norm(t-temp_table,axis=-1)
        index=tf.argmin(dist,axis=-1)
        y=tf.gather(self.lookup_table,index)
        return y,index,dist
    def trans_embed(self,x):
        t=tf.expand_dims(x,-2)
        temp_table=tf.reshape(self.lookup_table,[1,self.classes,self.hidden_size])
        dist=tf.norm(t-temp_table,axis=-1)
        index=tf.argmin(dist,axis=0)
        y=tf.gather(x,index)
        return y
    def assign(self,x):
        temp=tf.reduce_mean(x,axis=0)
        return tf.assign(self.lookup_table,self.lookup_table*0.999+0.0001*temp)

