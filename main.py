### PACKAGES
import csv
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import pickle
from copy import deepcopy
from asift import *
#import keras
from tensorflow import keras #for kelvin's machine
from find_obj import init_feature, filter_matches, explore_match
import time
### CONSTANTS
ROOTPATH = './'
IMG_HEIGHT = 490
IMG_WIDTH = 680
NUM_CHANNELS = 3
train_img_names = [] # list of train image names as strings
train_img_labels = [] # list of train images labels as tuples (latitude, longitude)
with open('COMP90086_2021_Project_train/train.csv') as train_csv:
    reader = csv.reader(train_csv, delimiter=',')
    next(reader) # skip header row
    for row in reader:
        name = row[0] # string
        label = (float(row[1]), float(row[2])) # tuple
        train_img_names.append(name)
        train_img_labels.append(label)

# train images stored as numpy array
resize = 0.5
IMG_HEIGHT = int(IMG_HEIGHT * resize)
IMG_WIDTH = int(IMG_WIDTH * resize)
train_size = (len(train_img_names), IMG_HEIGHT, IMG_WIDTH,NUM_CHANNELS )
#print(train_images.shape)


train_images = np.zeros(train_size, dtype='uint8')
for i in range(len(train_img_names)):
    name = train_img_names[i]
    subpath = 'COMP90086_2021_Project_train/train/' + name + '.jpg'
    img = cv2.imread(os.path.join(ROOTPATH, subpath))
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
#     print(img.shape)
    train_images[i] = img

# train labels stored as numpy array
train_labels = np.array(train_img_labels)
print(train_labels[0])
print(train_images.shape)
test_img_names = [] # list of test image names as strings
with open('COMP90086_2021_Project_test/imagenames.csv') as test_csv:
    reader = csv.reader(test_csv, delimiter=',')
    next(reader) # skip header row
    for row in reader:
        name = row[0] # string
        test_img_names.append(name)
        
# test images stored as numpy array
test_size = (len(test_img_names), IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS)
test_images = np.zeros(test_size, dtype='uint8')
for i in range(len(test_img_names)):
    name = test_img_names[i]
    subpath = 'COMP90086_2021_Project_test/test/' + name + '.jpg'
    img = cv2.imread(os.path.join(ROOTPATH, subpath))
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))

    
    test_images[i] = img
    
# train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
# test_images = test_images.reshape(test_images.shape[0], 32, 32, 3).astype('float32')
k_nn = 2
lowe_ratio = 0.7
num_closest_matches = 5

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# If there exists a checkpoint, use saved weights and indices
# if os.path.exists('./checkpoint/weights.pickle') and os.path.exists('./checkpoint/indices.pickle'):
#     with open(os.path.join("checkpoint","weights.pickle"),'rb') as handle:
#         weights = pickle.load(handle)
#     with open(os.path.join(os.getcwd(),"checkpoint","indices.pickle"),'rb') as handle:
#         indices = pickle.load(handle)
        
#     # Skip already completed test images
#     test_images_sift = test_images[len(weights):]
#     print("Found checkpoint, skipping to test image",len(weights))
# else:
weights = []
indices = []
    


for i in range(len(test_images)):    
    print(i)

    test_image = test_images[i]
    # convert test image to grayscale
    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    akaze  = cv2.AKAZE_create()

#     sift = cv2.SIFT_create() # initialise SIFT detector
    
    # SIFT keypoints and descriptors for test image
    keypoints_test, descriptors_test = akaze.detectAndCompute(gray_test_image, None)
#     print(descriptors_test)
        
    # initialise FLANN
    flann = cv2.BFMatcher(cv2.NORM_HAMMING)

    
    # stores 5 highest number of keypoint matches
    best_num_matches = np.zeros(num_closest_matches, dtype='uint64')
    
    matches_index = np.zeros(num_closest_matches,dtype='uint64')
    # stores 5 train images with highest number of keypoint matches
    #best_image_matches = np.zeros((num_closest_matches, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype='uint8')
    # stores  corresponding matching images 
    #circles_and_lines = np.zeros((num_closest_matches, IMG_HEIGHT, IMG_WIDTH*2, NUM_CHANNELS), dtype='uint8')

    for j in range(len(train_images)):
        train_image = train_images[j]
        # convert train image to grayscale
        gray_train_image = cv2.cvtColor(train_image,cv2.COLOR_BGR2GRAY)
        
        # SIFT keypoints and descriptors for train image
        keypoints_train, descriptors_train = akaze.detectAndCompute(gray_train_image, None)
        
        
        
        # when there is no descriptor less than k_nn descriptors
        if descriptors_test is None or descriptors_train is None or \
        descriptors_test.shape[0] < k_nn or descriptors_train.shape[0] < k_nn:
            continue
            
        # Draws a circle with the size of each keypoint and show its orientation
        keypoints_test_with_size = cv2.drawKeypoints(gray_test_image, keypoints_test, None,
                                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints_train_with_size = cv2.drawKeypoints(gray_train_image, keypoints_train, None,
                                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Matching descriptors using KNN algorithm
        matches = flann.knnMatch(descriptors_test, descriptors_train, k=k_nn)
        
        # Create a mask to draw all good matches
        matchesMask = []
        # Store all good matches as per Lowe's Ratio test.
        good = []
        for m,n in matches:
            if m.distance < lowe_ratio *n.distance:
                good.append(m)
                matchesMask.append([1,0]) # Match
            else:
                matchesMask.append([0,0]) # Mismatch
        num_matches = len(good)
        
        # Draw all good matches
        draw_params = dict(#matchColor = (0,255,0),  #If you want a specific colour
                           #singlePointColor = (255,0,0), #If you want a specific colour
                            matchesMask = matchesMask,
                            flags = cv2.DrawMatchesFlags_DEFAULT)

        good_matches = cv2.drawMatchesKnn(gray_test_image, keypoints_test, gray_train_image,
                                          keypoints_train, matches, None, **draw_params)
        
        # compares to closest matches and update as necessary
        for k in range(num_closest_matches):
            if num_matches >= best_num_matches[k]:
                best_num_matches = np.insert(best_num_matches, k, num_matches, 0)
                best_num_matches = np.delete(best_num_matches, -1, 0)
                matches_index = np.insert(matches_index, k, j, 0)
                matches_index = np.delete(matches_index, -1, 0)
                #best_image_matches = np.insert(best_image_matches, k, train_image, 0)
                #best_image_matches = np.delete(best_image_matches, -1, 0)
                #circles_and_lines = np.insert(circles_and_lines, k, good_matches, 0)
                #circles_and_lines = np.delete(circles_and_lines, -1, 0)
                break
    
    '''
    # display 5 closest images
    for z in range(num_closest_matches):
        plt.subplots(figsize=(10, 10)) 
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB), cmap='gray')  
        plt.title('Test Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(best_image_matches[z], cv2.COLOR_BGR2RGB), cmap='gray')  
        plt.title('Train Image')
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(15, 15))
        plt.imshow(circles_and_lines[z])
        plt.title(f'Number of good matches between two images: {best_num_matches[z]}')
        plt.axis('off')
        plt.show()
    '''
    weights.append(list(best_num_matches))
    indices.append(list(matches_index))
    
    if i % 5 == 0: # Save weights and Indices every 5 test images 
        with open(os.path.join("checkpoint","akaze_weights.pickle"),'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(os.getcwd(),"checkpoint","akaze_indices.pickle"),'wb') as handle:
            pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    