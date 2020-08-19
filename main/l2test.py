import numpy as np 
from scipy.spatial import distance

a = [1,2,3,4]
na = np.array(a)
print('l2-a',np.linalg.norm(na))
lna = na/np.linalg.norm(na)
print('lna',lna)

b = [4,3,2,1]
nb = np.array(b)
print('l2-b',np.linalg.norm(nb))
lnb = nb/np.linalg.norm(nb)
print('lnb',lnb)


c = [a,b]
nc = np.array(c)
print('nc.shape',nc.shape)
print('l2-c',np.linalg.norm(nc,axis=1,ord=2))
c_l2 = np.linalg.norm(nc,axis=1)
print('mm',nc/c_l2.reshape(-1,1))

print(distance.euclidean(na,na))
print(distance.euclidean(na,nb))

print(distance.euclidean(lna,lna))
print(distance.euclidean(lna,lnb))

'''d = np.random.random((1,64))
norm_d = np.linalg.norm(d,axis=1)'''

print('next')
'''import faiss
import numpy as np

a = np.array([1,2,3,4]).astype('float32')
a = a.reshape(1,-1)
p = np.array([9]).astype('float32')
print('dt',a.dtype,p.dtype)

d=4
i = faiss.IndexFlatL2(d)
print('1',i.is_trained)
i = faiss.IndexIDMap(i)

i.add_with_ids(a,p)'''
