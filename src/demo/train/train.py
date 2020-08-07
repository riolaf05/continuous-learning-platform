import tensorflow as tf
from tensorflow import keras
import mlflow
from mlflow import tensorflow
import os

BASE_DIR='/train/data'
testing = False
epochs = 1
version = 1

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))

model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])

#set MLflow server
tracking_uri="http://1a420b445ed3.ngrok.io"
mlflow.set_tracking_uri(tracking_uri)

#Set experiment
if mlflow.get_experiment_by_name("test") != None:
    exp_id = mlflow.set_experiment("test")
else: 
    exp_id = mlflow.create_experiment("test")

#Set tags
tags={}
tags['name']="test"

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_id=None, experiment_id=exp_id, run_name=None, nested=False):

    mlflow.tensorflow.autolog()

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    #save model
    #export_path = os.path.join(BASE_DIR, str(version))
    #print('export_path = {}\n'.format(export_path))
    #model.save("/train/data/models/model.pb") 

    #mlflow.log_param("type", "test")
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_acc", test_acc)
    mlflow.set_tags(tags)

    artifact_uri=mlflow.get_artifact_uri()
    print(artifact_uri)
    mlflow.log_artifacts(artifact_uri)
    
mlflow.end_run()