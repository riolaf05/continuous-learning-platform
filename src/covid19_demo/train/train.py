import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 
import tensorflow as tf
from tensorflow.math import confusion_matrix
import argparse
from numpy import loadtxt
import mlflow
from mlflow import tensorflow
from tensorflow.keras.optimizers import Adam

def main():
    parser = argparse.ArgumentParser(description='Input arguments')
    parser.add_argument('--img-size', type=int, help='Image size', default=200)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=16)
    parser.add_argument('--tracking-url', type=str, help='MLFlow server')
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    x = [] #features
    y = [] #target

    BASE_DIR='/train/data'
    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE=args.batch_size

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

    #converto le liste in array numpy
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

    #set MLflow server
    mlflow.set_tracking_uri(args.tracking_url)

    #Set experiment
    if mlflow.get_experiment_by_name(args.experiment) != None:
        exp_id = mlflow.set_experiment(args.experiment)
    else: 
        exp_id = mlflow.create_experiment(args.experiment)

    #Set tags
    tags={}
    tags['name']=args.experiment
    mlflow.set_tags(tags)

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_id=None, experiment_id=exp_id, run_name=None, nested=False):

        history = model.fit(train_generator, epochs=args.epochs, steps_per_epoch=x_train.shape[0]/BATCH_SIZE)

        acc = history.history['acc'][-1]
        loss = history.history['loss'][-1]

        metrics_train = model.evaluate(x_train, y_train)
        metrics_test = model.evaluate(x_test, y_test)

        print("Train accuracy =  %.4f - Train loss= %.4f" % (metrics_train[1], metrics_train[0]))
        print("Test accuracy =  %.4f - Test loss= %.4f" % (metrics_test[1], metrics_test[0]))

        mlflow.tensorflow.autolog()

        mobilenet_save_path = os.path.join(BASE_DIR, "models")
        tf.saved_model.save(model, mobilenet_save_path)

        mlflow.tensorflow.log_model(model, "model", registered_model_name="TestModel")

        ### Validate
        #y_predict =  model.predict_classes(x_test)

        #cm = confusion_matrix(y_test, y_predict) #primo parametro: predizioni corrette, secondo predizioni fatte da noi

        #df_cm = pd.DataFrame(cm, index=["Predicted Covid", "Predicted Normal"], columns=["Covid", "Normal"])

        #sn.heatmap(df_cm, annot=True)
    mlflow.end_run()

if __name__ == "__main__":
    main()