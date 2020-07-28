import os
import pandas as pd
import numpy as np
import cv2
import matplotlib as plt
import seaborn as sn
from shutil import copyfile
import matplotlib.pyplot as plt
from random import shuffle

BASE_DIR='/preprocess/data'

def copy_sample(imgs_list, imgs_path, cls):
  for img in imgs_list:
    copyfile(imgs_path+img, os.path.join(BASE_DIR, 'dataset/'+cls+"/"+img))

def show_samples(x):
  fig = plt.figure()
  for i in range(x.shape[0]):
    plot = fig.add_subplot(1, x.shape[0], i+1)
    plt.imshow(x[i])
    plt.axis("off")

def main():

  if(not os.path.isdir(os.path.join(BASE_DIR, "dataset"))):
    os.mkdir(os.path.join(BASE_DIR, "dataset/"))
    os.mkdir(os.path.join(BASE_DIR, "dataset/covid"))
    os.mkdir(os.path.join(BASE_DIR, "dataset/normal"))

  #radiografie pazienti con covid-19
  df = pd.read_csv(os.path.join(BASE_DIR, "covid-chestxray-dataset/metadata.csv"))
  df = df[(df["finding"]=="COVID-19") & (df["view"]=="PA")]
  imgs_covid=list(df["filename"])
  imgs_covid_count=len(imgs_covid)
  copy_sample(imgs_covid, os.path.join(BASE_DIR, "covid-chestxray-dataset/images/"), "covid")

  #radiograzie pazienti con e senza covid-19
  imgs_normal = os.listdir(os.path.join(BASE_DIR, "chest_xray/train/NORMAL/"))
  shuffle(imgs_normal)
  imgs_normal = imgs_normal[:imgs_covid_count]
  copy_sample(imgs_normal, os.path.join(BASE_DIR, "chest_xray/train/NORMAL/"), "normal")

if __name__ == "__main__":
    main()