import numpy as np
import cv2
import pandas as pd
import math 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.io import imread_collection
import glob
from IPython.display import Image
from os import *
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from numpy.linalg import norm

file = "./dataset"
images = []
c=0
for i in listdir(file):
    if (i.endswith(".png")):
        tmp = cv2.imread("./dataset/"+i,0).astype(np.uint8)
        images.append(tmp)
        plt.imshow(images[c],cmap="gray")
        plt.show()
        c+=1


extractor = cv2.SIFT_create()
keypoints = []
descriptors = []

for img in images:
    img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
    keypoints.append(img_keypoints)
    descriptors.append(img_descriptors)

all_descriptors = []
for img_descriptors in descriptors:
    for descriptor in img_descriptors:
        all_descriptors.append(descriptor)

all_descriptors = np.stack(all_descriptors)


## Plot graph
variances = []
k_values = range(100, 1000,100)
for k in k_values:
    _, variance = kmeans(all_descriptors, k, 1)
    variances.append(variance)

plt.plot(k_values, variances)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Variance")
plt.show()


k = 500
iters = 1
codebook, variance = kmeans(all_descriptors, k, iters)
print(variance)


visual_words = []
for img_descriptors in descriptors:
    img_visual_words, distance = vq(img_descriptors, codebook)
    visual_words.append(img_visual_words)


frequency_vectors = []
for i in visual_words:
    a,b,c = plt.hist(i, 20)
    plt.show()
    frequency_vectors.append(a)


q_file = "./query"
q_images = []
c=0
for i in listdir(q_file):
    tmp = cv2.imread("./query/"+i,0).astype(np.uint8)
    q_images.append(tmp)
    plt.imshow(q_images[c],cmap="gray")
    plt.show()
    c+=1
    


keypoints_1 = []
descriptors_1 = []

for img in q_images:
    img_keypoints, img_descriptors = extractor.detectAndCompute(img, None)
    keypoints_1.append(img_keypoints)
    descriptors_1.append(img_descriptors)



visual_words_1 = []
for img_descriptors in descriptors_1:
    img_visual_words, distance = vq(img_descriptors, codebook)
    visual_words_1.append(img_visual_words)


frequency_vectors_1 = []
for i in visual_words_1:
    a,b,c = plt.hist(i, 20)
    plt.show()
    frequency_vectors_1.append(a)



for k in range(len(frequency_vectors_1)):
    cosine_similarity = []
    for i in range(len(frequency_vectors)):
        temp = np.dot(frequency_vectors_1[k], frequency_vectors[i].T)/(np.linalg.norm(frequency_vectors[i]) * np.linalg.norm(frequency_vectors_1[k]))
        cosine_similarity.append(temp)
    idx = np.argsort(cosine_similarity)[::-1][:3]
    
    plt.imshow(q_images[k],cmap="gray")
    plt.title(f"QUERY Image {k+1}")
    plt.show()
    for i in idx:
        print(f"Match {i+1}: Cosine Similarity = {round(cosine_similarity[i], 4)}")
        plt.imshow(images[i],cmap="gray")
        plt.show()
















