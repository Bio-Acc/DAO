from tensorflow.examples.tutorials.mnist import input_data
import pickle
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq,whiten

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
y_test = mnist.test.labels


file=open('ae_embed_mnist_hidden.pkl','rb')
x_train=pickle.load(file)
x_test=pickle.load(file)
file.close()


#whitened=whiten(x_train)
codebook, distortion = kmeans(x_train, 10)

result=vq(x_test,codebook)

file=open('mnist_AE-kmeans_result.pkl','wb')
pickle.dump(codebook,file)
pickle.dump(distortion,file)
pickle.dump(result,file)
pickle.dump(y_test,file)
file.close()