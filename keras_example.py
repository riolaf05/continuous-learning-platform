import os
from numpy import loadtxt
import tensorflow as tf
import mlflow
from mlflow import tensorflow

BASE_DIR=r"C:\Users\lafacero\Desktop\mlflow test\\"
MODEL_DIR=r"C:\Users\lafacero\Desktop\mlflow test\models\\"
MLFLOW_MODEL_DIR=r"C:\Users\lafacero\Desktop\mlflow test\mlflow_models\\"

# load the dataset
dataset = loadtxt(BASE_DIR+'pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables

X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(12, input_dim=8, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

with mlflow.start_run():
        
    # fit the keras model on the dataset
    history = model.fit(X, y, epochs=10, batch_size=10)

    # evaluate the keras model
    acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    #val_loss = history.history['val_loss']

    mlflow.log_param("accuracy", acc)
    #mlflow.log_param("val_accuracy", val_acc)
    mlflow.log_param("loss", loss)
    #mlflow.log_param("val_loss", val_loss)

    mobilenet_save_path = os.path.join(BASE_DIR, "models")
    tf.saved_model.save(model, mobilenet_save_path)

    mlflow.tensorflow.log_model(model, "model", registered_model_name="TestModel")
    
    '''
    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="TestModel")
    else:
        mlflow.sklearn.log_model(lr, "model")
    '''