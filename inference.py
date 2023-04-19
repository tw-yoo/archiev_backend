import datetime
import io
from urllib import request
import numpy as np
from PIL import Image

from emission_result import LabelAndProb

Image.MAX_IMAGE_PIXELS = None
import tensorflow as tf
import tensorflow_hub as hub
from codecarbon import track_emissions, EmissionsTracker

from datetime import datetime

now = datetime.now()


def get_image(uid, r):
    tracker = EmissionsTracker(project_name="step1", output_file=f"result/{now}.csv")
    tracker.start()

    data_io = io.BytesIO(r)
    image = Image.open(data_io)

    tracker.stop()

    return image


def load_model(uid, model_url: str, input_size: int):
    tracker = EmissionsTracker(project_name="step2", output_file=f"result/{now}.csv")
    tracker.start()

    model = tf.keras.Sequential([
        hub.KerasLayer(model_url)
    ])
    model.build([None, input_size, input_size, 3])

    tracker.stop()

    return model


def get_resized_np_array(uid, image, input_size):
    tracker = EmissionsTracker(project_name="step3", output_file=f"result/{now}.csv")
    tracker.start()

    image = image.resize((480, 480))
    x = np.array(image)
    x = x / 255.0

    x = np.expand_dims(x, axis=0)

    tracker.stop()

    return x


def get_inference_result(uid, model, np_array):
    tracker = EmissionsTracker(project_name="step4", output_file=f"result/{now}.csv")
    tracker.start()

    probs = model.predict(np_array)
    probs_norm = np.squeeze(np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, probs))

    tracker.stop()

    return list(probs_norm)


def convert_inference_result(probs):

    a = open("label/image21k.txt")
    label_list = [
        i.strip()
        for i in a.readlines()
    ]

    result = {}
    for i in range(len(probs)):
        label = label_list[i]
        prob = probs[i]
        result[label] = prob

    label_and_prob_list = []
    for k in result.keys():
        label_and_prob_list.append(
            LabelAndProb(k, float(result[k])).__dict__
        )

    return label_and_prob_list


def get_model_url_and_size(model_name, model_df):

    model = model_df.query(f"name == '{model_name}'").iloc[0]

    return model['url'], model['input_size']


def get_inference_result_probs(uid, model_df, model_name, r):

    image = get_image(uid, r)

    model_url, input_size = get_model_url_and_size(model_name, model_df)

    model = load_model(uid, model_url, input_size)

    resized_np_array = get_resized_np_array(uid, image, input_size)

    probs = get_inference_result(uid, model, resized_np_array)

    return convert_inference_result(probs)
