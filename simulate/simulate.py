import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import pickle


#omics 1
data=[]
types=[]
for i in range(10):
    center=5*np.random.random(100)
    noise=np.random.normal(np.array([0]*100),size=(1000,100))
    d=np.random.normal(center,size=(1000,100))
    print(np.mean(d,axis=0)-center)
    d=d+noise
    print(d.shape)
    data.append(d)
    types.extend([i]*1000)

#omics 2
data1=[]
types1=[]
for i in range(2):
    center=2*np.random.random(500)
    noise=np.random.normal(np.array([0]*500),size=(5000,500))
    d=np.random.normal(center,size=(5000,500))
    d=d+3*noise
    data1.append(d)
    types1.extend([i]*5000)


data=np.concatenate(data)
data1=np.concatenate(data1)

bivec_1=TSNE(perplexity=50.0).fit_transform(data)
bivec_1=bivec_1.transpose()
bivec_2=TSNE(perplexity=50.0).fit_transform(data1)
bivec_2=bivec_2.transpose()

plt.figure()
cmap=[plt.get_cmap('tab20'),plt.get_cmap('tab20b')]
for i in range(10):
    index=np.arange(0+i*1000,1000+i*1000)
    color=i
    scatter=plt.scatter(bivec_1[0][index],bivec_1[1][index],c=cmap[i//20](color),s=1,label=i,alpha=1/3)
plt.legend(loc=2,markerscale=5,bbox_to_anchor=(1.0,0.,0.5,1.0),ncol=2,frameon=True)
plt.savefig('simulate_origin_1.png',dpi=500,bbox_inches = 'tight')
plt.close()

plt.figure()
cmap=[plt.get_cmap('tab20'),plt.get_cmap('tab20b')]
for i in range(2):
    index=np.arange(0+i*5000,5000+i*5000)
    color=i
    scatter=plt.scatter(bivec_2[0][index],bivec_2[1][index],c=cmap[i//20](color),s=1,label=i,alpha=1/3)
plt.legend(loc=2,markerscale=5,bbox_to_anchor=(1.0,0.,0.5,1.0),ncol=2,frameon=True)
plt.savefig('simulate_origin_2.png',dpi=500,bbox_inches = 'tight')
plt.close()

types1=np.array(types1)
types1=np.expand_dims(types1,0)
data1=data1.transpose()
data1=np.concatenate([data1,types1],axis=0)
data1=data1.transpose()
data1=np.random.permutation(data1)
types1=data1[:,-1]
data1=data1[:,:-1]
data=np.concatenate([data,data1],axis=1)
print(data.shape)
types=np.array(types)
bivec=TSNE(perplexity=50.0).fit_transform(data)
bivec=bivec.transpose()

plt.figure()
cmap=[plt.get_cmap('tab20'),plt.get_cmap('tab20b')]
for i in range(10):
    index=np.argwhere(types==i)
    color=i
    scatter=plt.scatter(bivec[0][index],bivec[1][index],c=cmap[i//20](color),s=1,label=i,alpha=1/3)
plt.legend(loc=2,markerscale=5,bbox_to_anchor=(1.0,0.,0.5,1.0),ncol=2,frameon=True)
plt.savefig('simulate_test.png',dpi=500,bbox_inches = 'tight')
plt.close()

types=types*2+types1
types=np.expand_dims(types,0)
data=data.transpose()
data=np.concatenate([data,types],axis=0)
data=data.transpose()
data=np.random.permutation(data)

x_train=data[:8000,:-1]
y_train=data[:8000,-1]
x_test=data[8000:,:-1]
y_test=data[8000:,-1]

file=open('simulate_data.pkl','wb')
pickle.dump(x_train,file)
pickle.dump(y_train,file)
pickle.dump(x_test,file)
pickle.dump(y_test,file)
file.close()
