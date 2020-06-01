import os
import csv
import cv2
import numpy as np
import keras
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

import sklearn
from sklearn.model_selection import train_test_split


# print GPU details ...
import GPUtil as GPU
GPUs = GPU.getGPUs()
gpu = GPUs[0]
print()
print("Total Available GPU memory: {} MB".format(gpu.memoryTotal))
print("Used GPU memory: {} MB".format(gpu.memoryUsed))
print("Total Free GPU memory: {} MB".format(gpu.memoryFree))
print()

# printing installed tensorflow/keras/OpenCV versions
print("Current OpenCV version", cv2.__version__)
print("Using TensorFlow version", tf.__version__)
print("Using Keras version", keras.__version__)

# 1. Start of dataset building
lines = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
directory = './my_data/IMG/'

# steering angle correction
correction = 0.22

# setting data augmentation activation to True/False
augment = True

if (augment):
    print("Performing data augmentation by flipping the images ...\n")

for line in lines:

    # reading filesnames for center, left and right images
    fname_center = directory + line[0].split('/')[-1]
    fname_left = directory + line[1].split('/')[-1]
    fname_right = directory + line[2].split('/')[-1]

    # reading images using opencv
    img_center = cv2.imread(fname_center)
    img_left = cv2.imread(fname_left)
    img_right = cv2.imread(fname_right)

    # adding images to the lists
    images.append(img_center)
    images.append(img_left)
    images.append(img_right)

    # adding steering angles
    measurements.append(float(line[3]))
    measurements.append(float(line[3])+correction)  # for left image
    measurements.append(float(line[3])-correction)  # for right image

    # only if data augmentation is required
    if (augment):
        # this is where I perform data augmentation for images and angles
        center_flipped = np.fliplr(img_center)
        images.append(center_flipped)
        center_measurement_flipped = -(float(line[3]))
        measurements.append(center_measurement_flipped)

        left_flipped = np.fliplr(img_left)
        images.append(left_flipped)
        left_measurement_flipped = -(float(line[3])+correction)
        measurements.append(left_measurement_flipped)

        right_flipped = np.fliplr(img_right)
        images.append(right_flipped)
        right_measurement_flipped = -(float(line[3])-correction)
        measurements.append(right_measurement_flipped)


# define arrays for datasets
X_train = np.array(images)
steering_angles = np.array(measurements)

# function to display some images
def disp_images():
    f, ax = plt.subplots(2, 3, figsize=(50, 10))
    ax = ax.ravel()
    for i in range(6):
        index = random.randint(0, len(X_train))
        image = X_train[index]
        ax[i].imshow(image)
        plt.axis('off')
        ax[i].set_title("steering angle - " + str(steering_angles[index]), fontsize=20)

    # saving the figure
    plt.savefig('markdown_images/sample_images.jpg')


def print_dataset_info():

    if (augment):
        # print out array shapes and details
        print("Original Dataset information ---\n")
        # print("Image shape: ", X_train[0][1])
        print("Steering Angles:", 0.5*len(steering_angles))
        print("Total images:", 0.5*X_train.shape[0])
        print()

        print("Augmented Dataset information ---\n")
        # print("Image shape: ", X_train[0][1])
        print("Steering Angles after augmentation:", len(steering_angles))
        print("Total images after augmentation:", X_train.shape[0])
        print()
    else:
        # print out array shapes and details
        print("Original Dataset information ---\n")
        # print("Image shape: ", X_train[0][1])
        print("Steering Angles:", 0.5*len(steering_angles))
        print("Total images:", 0.5*X_train.shape[0])


# uncomment this section to print information on dataset
print_dataset_info()
disp_images()

'''
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col)))

model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))

# 1st Convolutional layer
model.add(Conv2D(24, (5, 5), subsample = (2,2), activation="relu"))
# 2nd Convolutional layer
model.add(Conv2D(36, (5, 5), subsample = (2,2), activation="relu"))
# 3rd Convolutional layer
model.add(Conv2D(48, (5, 5), subsample = (2,2), activation="relu"))
# 4th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# 5th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# 6th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Dropout(0.5))

# flatten layers into a vector
model.add(Flatten())

# four fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')

# fit model with a validation set of 20%
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)
'''


# 2. Start of Network Architecture
# NVIDIA network architecture
model = Sequential()

# image preprocessing - normalizing the pixel values and cropping the image
model.add(Lambda(lambda x:(x/255.0)-0.5, input_shape=(160,320,3)))
# cropping top 50 pixels and 20 pixels from the bottom
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))

# 1st Convolutional layer
model.add(Conv2D(24, (5, 5), subsample = (2,2), activation="relu"))
# 2nd Convolutional layer
model.add(Conv2D(36, (5, 5), subsample = (2,2), activation="relu"))
# 3rd Convolutional layer
model.add(Conv2D(48, (5, 5), subsample = (2,2), activation="relu"))
# 4th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# 5th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))
# 6th Convolutional layer
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Dropout(0.5))

# flatten layers into a vector
model.add(Flatten())

# four fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')

# fit model with a validation set of 20%
model.fit(X_train, steering_angles, validation_split=0.2, shuffle=True, epochs=10)
model.save('cnn_model.h5')

model.summary()

