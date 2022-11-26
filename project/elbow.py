import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils


from sklearn import metrics
from scipy.spatial.distance import cdist


clusters = 5 # try changing it
img = cv2.imread('image.png')
org_img = img.copy()
print('Org image shape --> ',img.shape)
img = imutils.resize(img,height=200)
print('After resizing shape --> ',img.shape)
flat_img = np.reshape(img,(-1,3))
print('After Flattening shape --> ',flat_img.shape)



distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
  


for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(flat_img)
    kmeanModel.fit(flat_img)
  
    distortions.append(sum(np.min(cdist(flat_img, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / flat_img.shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(flat_img, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / flat_img.shape[0]
    mapping2[k] = kmeanModel.inertia_


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

