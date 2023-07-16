# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:54:57 2023

@author: Ivan
"""
import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import imghdr
from skimage.measure import label, regionprops
import random
import time


PATH="original"

def find_images(dirpath):
    """Recursively detect image files contained in a folder.

    Parameters
    ----------
    dirpath : string
        Path name of the folder that contains the images.

    Returns
    -------
    imgfiles : list
        Full path names of all the image files in `dirpath` (and its
        subfolders).

    Notes
    -----
    The complete list of image types being detected can be found at:
    https://docs.python.org/3/library/imghdr.html
    
    """
    imgfiles = [os.path.join(root, filename)
                for root, dirs, files in os.walk(dirpath)
                for filename in files
                if imghdr.what(os.path.join(root, filename))]
    
    return imgfiles

def get_class(filename):
    """Extract the class label from the path of a Kather dataset sample.
    
    Parameters
    ----------
    filename : string
        Filename (including path) of a texture sample 
        from Kather dataset.

    Returns
    -------
    class_name : string
        Class name to which the texture sample belongs.
    
    """
    folder, _ = os.path.split(filename)
    _, class_name = os.path.split(filename)
    class_name= class_name.split('-')
    
    return class_name


# Load image data and labels
data=[np.asarray(Image.open(path)) for path in find_images(PATH)]
labels=[get_class(clasa)[0] for clasa in find_images(PATH)]

#Create list of features for each imag 
regionProps=[regionprops(image)[0] for image in data]
imageFeautures=[[image_regionprops.eccentricity,image_regionprops.area_convex,image_regionprops.perimeter,image_regionprops.solidity,image_regionprops.euler_number,image_regionprops.area_bbox,image_regionprops.equivalent_diameter_area, image_regionprops.orientation,image_regionprops.solidity] for image_regionprops in regionProps]


loo = LeaveOneOut()
scaler = StandardScaler()
image_features = np.array(img_features)
labels = np.array(labels)


image_features = scaler.fit_transform(image_features)

X_train, X_test, y_train, y_test = train_test_split(imageFeautures, labels, test_size=0.2, random_state=42)

results=[]

# Making prediction for range of 1 to 9 Neighbors 
for i in range(1,10):
    classifier = KNeighborsClassifier(n_neighbors=i) 
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    print("Accuracy:",acc, ", for ",i, " Neigbors.")
    results.append(acc)

def display_random_image(dataset):
    randInt=random.randint(0, len(dataset))
    plt.imshow(data[imageFeautures.index(dataset[randInt])])
    y_pred = classifier.predict(dataset[randInt-1:randInt+1])
    time.sleep(1)
    print('Predicted class:',y_pred[1])

#Function will display 9 Neighbors Classifier
display_random_image(X_test)



plt.rcParams["figure.figsize"] = (10, 7)
plt.plot(results)
plt.xticks(range(9), range(1,10))
plt.xlabel("number of neighbors")
plt.ylabel("Accuracy")





