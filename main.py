import os
import pandas as pd
import numpy as np
import cv2
import matplotlib as plt
import seaborn as sn
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 
import tensorflow as tf
from tensorflow.math import confusion_matrix
import mlflow
from random import shuffle

BASE_DIR=r"C:\Users\lafacero\Desktop\covid-mlflow"
IMG_SIZE = (200, 200)
BATCH_SIZE=16

x = [] #features
y = [] #target

def copy_sample(imgs_list, imgs_path, cls):
  for img in imgs_list:
    copyfile(imgs_path+img, os.path.join(BASE_DIR, 'dataset/'+cls+"/"+img))

def show_samples(x):
  fig = plt.figure()
  for i in range(x.shape[0]):
    plot = fig.add_subplot(1, x.shape[0], i+1)
    plt.imshow(x[i])
    plt.axis("off")

if(not os.path.isdir(os.path.join(BASE_DIR, "dataset"))):
  os.mkdir(os.path.join(BASE_DIR, "dataset/"))
  os.mkdir(os.path.join(BASE_DIR, "dataset/covid"))
  os.mkdir(os.path.join(BASE_DIR, "dataset/normal"))

df = pd.read_csv(os.path.join(BASE_DIR, "covid-chestxray-dataset/metadata.csv"))

df = df[(df["finding"]=="COVID-19") & (df["view"]=="PA")]

imgs_covid=list(df["filename"])

imgs_covid_count=len(imgs_covid)

copy_sample(imgs_covid, os.path.join(BASE_DIR, "covid-chestxray-dataset/images/"), "covid")

imgs_normal = os.listdir(os.path.join(BASE_DIR, "chest_xray/train/NORMAL/"))

shuffle(imgs_normal)
imgs_normal = imgs_normal[:imgs_covid_count]

copy_sample(imgs_normal, "chest_xray/train/NORMAL/", "normal")

encoding = [("normal", 0), ("covid", 1)]

for folder, label in encoding: 
  current_folder=os.path.join(BASE_DIR, "dataset/"+folder+"/")
  for img_name in os.listdir(current_folder):
    img = cv2.imread(current_folder+img_name, cv2.IMREAD_GRAYSCALE)
    
    #uso la normalizzazione dell'istrogramma per assicurarmi che la differenza nella luminosità dell'immagine 
    #non creino problemi
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, IMG_SIZE)

    #normalizzo i pixel per portare tutti i pixel da [0, 255] a [0, 1]
    img=img/255

    x.append(img)
    y.append(label)

#converto le liste in arrya numpy
x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_test.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

datagen = ImageDataGenerator(
    rotation_range=15,
    brightness_range=[0.2, 1.0]
)

train_generator = datagen.flow(
    x_train,
    y_train, 
    batch_size = BATCH_SIZE #numero di immagini che il generatore genererà ad ogni iterazione
)

### Train
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[0], 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.6))
model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train_generator, epochs=100, steps_per_epoch=x_train.shape[0]/BATCH_SIZE)

metrics_train = model.evaluate(x_train, y_train)
metrics_test = model.evaluate(x_test, y_test)

print("Train accuracy =  %.4f - Train loss= %.4f" % (metrics_train[1], metrics_train[0]))
print("Test accuracy =  %.4f - Test loss= %.4f" % (metrics_test[1], metrics_test[0]))

### Validate
y_predict =  model.predict_classes(x_test)

cm = confusion_matrix(y_test, y_predict) #primo parametro: predizioni corrette, secondo predizioni fatte da noi

df_cm = pd.DataFrame(cm, index=["Predicted Covid", "Predicted Normal"], columns=["Covid", "Normal"])

sn.heatmap(df_cm, annot=True)