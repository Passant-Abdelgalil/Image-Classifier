from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub



def process_image(image):
    test_image = np.asarray(image)
    tensor_image = tf.convert_to_tensor(test_image)
    resized_image = tf.image.resize(tensor_image, (224,224))/255
    return np.expand_dims(resized_image.numpy(), axis=0)


def predict(img, model, top_k=1):
    probs = model.predict(img)[0]
    classes_indecies = [ str(ind+1) for ind in np.argsort(probs)[-top_k:]]
    return probs[np.argsort(probs)[-top_k:]], classes_indecies


def read_image(file_path):
    return Image.open(file_path)


def load_model(model_file):
    return tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)


def load_jsonify_classes(file_name):
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    return class_names

