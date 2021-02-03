import pickle
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq,whiten

file=open('../stimulate_data.pkl','rb')
_=pickle.load(file)
y_train=pickle.load(file)
_=pickle.load(file)
y_test=pickle.load(file)
file.close()



file=open('ae_stimulate_hidden.pkl','rb')
x_train=pickle.load(file)
x_test=pickle.load(file)
file.close()

x_train=x_train[0]
x_test=x_test[0]


#whitened=whiten(x_train)
codebook, distortion = kmeans(x_train, 20)

result=vq(x_test,codebook)

file=open('simulate_AE-kmeans_result.pkl','wb')
pickle.dump(codebook,file)
pickle.dump(distortion,file)
pickle.dump(result,file)
pickle.dump(y_test,file)
file.close()

