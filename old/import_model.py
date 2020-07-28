import mlflow.tensorflow
import tensorflow as tf

MODEL_URI=r"C:\Users\lafacero\Desktop\mlflow test\models\saved_model.pb"

tf_graph = tf.Graph()
#tf_sess = tf.Session(graph=tf_graph)

with tf_graph.as_default():
    signature_definition = mlflow.tensorflow.load_model(model_uri=MODEL_URI,
                            tf_sess=None)

print(signature_definition)