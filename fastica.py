import numpy as np
import sklearn.decomposition
import pickle
from sklearn.decomposition import FastICA

print('loading data')
with np.load("/media/tbell/sanborn/rd_analysis/inputs/vh_test_ftwhite.npz") as d:
    data = d['arr_0'].item()
    
version = 'v1'
print('beginning training')
# K, W, S = sklearn.decomposition.fastica(data['train'].images, whiten=True, tol=.001, max_iter=1000)

ica = FastICA(whiten=False, tol=.0001, max_iter=20000)
ica.fit(data['train'].images)
filters = ica.components_
print('training complete')

# pickle.dump(K, open( "/media/tbell/sanborn/rd_analysis/outputs/fastica/K_"+version+".p", "wb" ))
# pickle.dump(W, open( "/media/tbell/sanborn/rd_analysis/outputs/fastica/W_"+version+".p", "wb" ))
# pickle.dump(S, open( "/media/tbell/sanborn/rd_analysis/outputs/fastica/S_"+version+".p", "wb" ))

pickle.dump(filters, open( "/media/tbell/sanborn/rd_analysis/outputs/fastica/weights_"+version+".p", "wb" ))


# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
 
# from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import FastICA
 
# # fetch natural image patches
# image_patches = fetch_mldata("natural scenes data")
# X = image_patches.data
 
# # 1000 patches a 32x32
# # not so much data, reshape to 16000 patches a 8x8
# X = X.reshape(1000, 4, 8, 4, 8)
# X = np.rollaxis(X, 3, 2).reshape(-1, 8 * 8)
 
# # perform ICA
# ica = FastICA(n_components=49)
# ica.fit(X)
# filters = ica.components_

# pickle.dump(filters, open( "/media/tbell/sanborn/rd_analysis/outputs/fastica/test_weights.p", "wb" ))
