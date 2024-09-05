# Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset. 
# Dataset :- https://www.kaggle.com/c/dogs-vs-cats/data

import numpy as np 
import random
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten
from tensorflow.keras.callbacks import TensorBoard
import time

Directory = r'C:\\Users\\riyaz\\Desktop\\Prodigy Infotech\\Prodigy-ML-03\\train'
Categories = ['cat','dog']


Data = []
img_size = 100
for category in Categories:
    folder = os.path.join(Directory,category)
    label = Categories.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(img_size,img_size))
        Data.append([img_arr,label])
        

random.shuffle(Data)


X = []
y = []

for features , labels in Data:
    X.append(features)
    y.append(labels)

X = np.array(X)
y = np.array(y)

Name = f'cat-dog-prediction-{int(time.time())}'

tensorBoard = TensorBoard(log_dir=f'logs\\{Name}\\')

X = X/255

model = Sequential()

model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3), activation = "relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape = X.shape[1:], activation = "relu"))
model.add(Dense(128, activation = "relu"))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

model.fit(X,y,epochs=5,validation_split = 0, batch_size = 32 , callbacks = [tensorBoard])

Y_test = np.loadtxt('sampleSubmission.csv', delimiter=",")

id2 = random.randint(0, len(Y_test))

plt.imshow(X[id2, :])
plt.show()

y_pred = model.predict (X[id2, : ].reshape(1,100,100,3))
print(y_pred)