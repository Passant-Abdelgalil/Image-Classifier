from PIL import Image
import json
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
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
    try:
        image = Image.open(file_path)
        return image
    except Exception:
        print("\n========= not a valid image file path! ==========")
        return None


def load_model(model_file):
    try:
        model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
        return model
    except Exception:
        print("\n========= not a valid path for a .h5 file =========")
        return None


def load_jsonify_classes(file_name):
    try:
        with open(file_name, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception:
        print("\n========= Not a valid JSON file ==========")
        return None

