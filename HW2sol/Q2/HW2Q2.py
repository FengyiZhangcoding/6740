"""
this script is for 6740 2021Summer HW2-Q2

@author: PFC
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist

from sklearn.utils.graph_shortest_path import graph_shortest_path
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# select LP norm
lp = 1
    

# Loading Data
images = loadmat('isomap.mat')['images']
(d, n) = np.shape(images)


# Generate Matrix for Similarity Graph
A = np.zeros([n,n])
if lp==1:
    A = cdist(images.T, images.T, 'cityblock')
else:
    A = cdist(images.T, images.T, 'euclidean')


# ### espilon-ISOMAP
Alist =np.sort(A.reshape(-1,)) # sorted all pair-wise distance

idx = n+40000
epsilon = Alist[idx] # the first n distances corrspond to the diagonal, i.e., the distance to the node itself, thus will be zero

plt.figure()
plt.plot(Alist)
plt.scatter(idx, epsilon, marker='x', c='r',label='threshold={:.4f}'.format(epsilon))
plt.title('sorted pair-wise Eulidean distance, Lp{}'.format(lp))
plt.legend(loc='lower right')
# plt.savefig('sorted-dist-lp{}.pdf'.format(lp))
plt.show()


# # find the epsilon nearest neighbor graph and enforce the symmetric matrix  
B = A<epsilon  
B = B|B.T     
B = B.astype('float')

print('threshold value: {:.4f}'.format(epsilon))
print('edges preserved: {}'.format(B.sum()))

G = A*B +99999.9*(1-B) # ## adding large number to indicate non-edge

#%% plot the adjacency matrix
# Plot the Adjacency Matrix by Intensity


fig_graph = plt.figure(constrained_layout=True)

gs_graph = gridspec.GridSpec(ncols=4, nrows=3, figure=fig_graph)

# ## show graph as an image
ax_graph = fig_graph.add_subplot(gs_graph[:,:3])
ax_graph.imshow(G,cmap=plt.get_cmap('gray'), extent=[0,698, 0, 698])
ax_graph.set_aspect('equal')

# ## randomly select some image
selected_random_img= np.array([128, 300, 400]) # image index

# ## display images
ax_graph.scatter(selected_random_img, 698-selected_random_img, marker='x',c='r')

img_graph1 = np.reshape(images[:,selected_random_img[0]], [64, -1]).T
img_graph2 = np.reshape(images[:,selected_random_img[1]], [64, -1]).T
img_graph3 = np.reshape(images[:,selected_random_img[2]], [64, -1]).T

ax_graph1 = fig_graph.add_subplot(gs_graph[0,3])
ax_graph2 = fig_graph.add_subplot(gs_graph[1,3])
ax_graph3 = fig_graph.add_subplot(gs_graph[2,3])

ax_graph1.imshow(img_graph1, cmap=plt.get_cmap('gray'))
ax_graph1.axis('off')
ax_graph2.imshow(img_graph2, cmap=plt.get_cmap('gray'))
ax_graph2.axis('off')
ax_graph3.imshow(img_graph3, cmap=plt.get_cmap('gray'))
ax_graph3.axis('off')

fig_graph.suptitle('Weighted Adjacency Matrix, Lp{}'.format(lp))
# fig_graph.savefig('Weighted-Adjacency-Matrix-Lp{}.pdf'.format(lp))

#%% embedding
D = graph_shortest_path(G)

# enforce symmetric due to the error fromeig the computing accuracy of the system
D = (D + D.T)/2

# Compute Matrix C
ones = np.ones([n,1])
H = np.eye(n) - 1/n*ones.dot(ones.T)
C = -H.dot(D**2).dot(H)/(2*n)  # here we normalize the magnitude by the nubmer of data point, this is only for the visualization purpose.

C = (C+C.T)/2
#print(np.max(C-C.T))
eig_val, eig_vec = np.linalg.eig(C)
eig_val = eig_val.real # to avoid numerical errs
eig_vec = eig_vec.real


eig_index = np.argsort(-eig_val) # Sort eigenvalue from large to small

# ## the 2-d embedding
Z = eig_vec[:,eig_index[0:2]].dot(np.diag(np.sqrt(eig_val[eig_index[0:2]])))


#%%
fig = plt.figure(constrained_layout=True)

gs = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)

# Plot Embedding
ax0 = fig.add_subplot(gs[:,:3])

index = np.argsort(Z[:,1])[4:7] # ## find three image close to each other

ax0.scatter(Z[:,0], Z[:,1], s = 5)
ax0.scatter(Z[index,0],Z[index,1], marker='x', c='red', label='selected image location')
ax0.set_title('Lp{}'.format(lp))
ax0.legend(bbox_to_anchor=(0.5, -0.2), loc='center', borderaxespad=0.1)
ax0.set_aspect('equal')

# ## display images
img_1 = np.reshape(images[:,index[0]], [64, -1]).T
img_2 = np.reshape(images[:,index[1]], [64, -1]).T
img_3 = np.reshape(images[:, index[2]], [64, -1]).T

ax1 = fig.add_subplot(gs[0,3])
ax2 = fig.add_subplot(gs[1,3])
ax3 = fig.add_subplot(gs[2,3])

ax1.imshow(eval('img_'+str(1)), cmap=plt.get_cmap('gray'))
ax1.axis('off')
ax2.imshow(eval('img_'+str(2)), cmap=plt.get_cmap('gray'))
ax2.axis('off')
ax3.imshow(eval('img_'+str(3)), cmap=plt.get_cmap('gray'))
ax3.axis('off')

# fig.savefig('lp{}-result.pdf'.format(lp))


#%%  show all faces on scatter plot

 
figx, axx= plt.subplots()
axx.scatter(Z[:,0], Z[:,1], s = 5)
axx.set_aspect('equal')

for ii in range(n):
    img = images[:,ii].reshape(64,64).T
    img = OffsetImage(img, cmap='gray', zoom=0.1) 
    axx.add_artist(AnnotationBbox(img, (Z[ii,0],Z[ii,1]),frameon=False))
axx.set_title('Lp{}'.format(lp))
# figx.savefig('all-images-lp{}.pdf'.format(lp),dpi=300)    



#%% with PCA
mu = images.mean(axis=1).reshape(-1,1)
demean = images - mu
v,s,_ = np.linalg.svd(demean)
Zpca = demean.T @ v[:,:2] 

figpca, axpca= plt.subplots()
axpca.scatter(Zpca[:,0], Zpca[:,1], s = 5)
axpca.set_aspect('equal')

for ii in range(n):
    img = images[:,ii].reshape(64,64).T
    img = OffsetImage(img, cmap='gray', zoom=0.1) 
    axpca.add_artist(AnnotationBbox(img, (Zpca[ii,0],Zpca[ii,1]),frameon=False))
axpca.set_title('PCA')
# figpca.savefig('all-images-pca.pdf',dpi=300)  

