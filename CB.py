

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import time
import cv2
from hist_ext import HistogramExtractor
import Similarity as sim

# initialize 
path = "C:\\Users\\Muhammed Elbouadi\\Desktop\\MyRestAI\\Resources\\"
data = "Dataset_images"
query = "22866_INiGLjOEdELLMkDwEcloeXFLnWG9cB5qCIHilQBg.jpg"
data_path = path+data
query_path = path+query



ima = os.listdir(data_path)
list = []
for x in ima:
    file =  data_path + "\\" +x
    print(file)
    image = io.imread(file)
    dim = (256, 256)
    resized_images = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized_images.shape)
    list.append(resized_images)
# --------->>>>
# --------->>>>

imagq = io.imread(query_path)
immg = np.array(imagq)

# load the data "images"

t = time.time()
print('Loading images...')
for root, dirs, files in os.walk(data_path, topdown=True):
    imgs = np.array([io.imread(os.path.join(data_path, file_name)) for file_name in files], dtype= object)
    tt = (time.time() - t) 

print (tt)
# --------->>>>
# ?????????????
# Features extraction  of data "images" by color
print('Extracting features...')
hist = HistogramExtractor(nbins_per_ch=(12, 8, 3), use_hsv=False, use_regions=False)
preds = np.array([hist.extract(img) for img in imgs], dtype = object)
ttt = (time.time() - t) 
print (ttt) 
# --------->>>>


# Features extraction of data "images" by forme "shape"
# --------->>>>


# Features extraction  of data "images" by texture
# --------->>>>



# load the query image
image_query = io.imread(query_path)
dim = (256, 256)
image_query = cv2.resize(image_query, dim, interpolation = cv2.INTER_AREA)
print('Resized Dimensions : ',image_query.shape)
immg = np.array(image_query)

# Finding similar images
print('Finding similar images...')
# --------->>>>

# Features extraction of query image
query_img = hist.extract(immg)
# --------->>>>

# loop over the results
sims = np.array([sim.cosine_similarity_1(query_img, other) for other in preds])
# --------->>>>

# Returns the indices that would sort an array
most_sim = np.argsort(sims)
# --------->>>>

# display the query
plt.title('Query')
plt.imshow(image_query)
plt.show()
# --------->>>>

    
# display the 1st simimlar image
plt.title('Result #1 (Sim: %.2f)' % sims[most_sim[-1]])
plt.imshow(imgs[most_sim[-1]])
plt.show()
# --------->>>>

# display the 2th simimlar image
plt.title('Result #2 (Sim: %.2f)' % sims[most_sim[-2]])
plt.imshow(imgs[most_sim[-2]])
plt.show()
# --------->>>>

# display the 3th simimlar image
plt.title('Result #3 (Sim: %.2f)' % sims[most_sim[-3]])
plt.imshow(imgs[most_sim[-3]])
plt.show()
# --------->>>>

# display the 4th simimlar image
plt.title('Result #4 (Sim: %.2f)' % sims[most_sim[-4]])
plt.imshow(imgs[most_sim[-4]])
plt.show()
# --------->>>>

# display the 5th simimlar image
plt.title('Result #5 (Sim: %.2f)' % sims[most_sim[-5]])
plt.imshow(imgs[most_sim[-5]])
plt.show()
# --------->>>>

tttt = (time.time() - t) 
print (tttt) 
