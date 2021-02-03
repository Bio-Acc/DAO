from tensorflow.examples.tutorials.mnist import input_data
import pickle
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq,whiten

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


codebook, distortion = kmeans(x_train, 10)

result=vq(x_test,codebook)

file=open('mnist_kmeans_result.pkl','wb')
pickle.dump(codebook,file)
pickle.dump(distortion,file)
pickle.dump(result,file)
pickle.dump(y_test,file)
file.close()
