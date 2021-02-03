import pickle
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq,whiten

file=open('simulate_data.pkl','rb')
x_train = pickle.load(file)
y_train = pickle.load(file)
x_test = pickle.load(file)
y_test = pickle.load(file)
file.close()



codebook, distortion = kmeans(x_train, 20)

result=vq(x_test,codebook)

file=open('simulate_kmeans_result.pkl','wb')
pickle.dump(codebook,file)
pickle.dump(distortion,file)
pickle.dump(result,file)
pickle.dump(y_test,file)
file.close()
