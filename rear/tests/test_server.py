import pytest
import requests
import base64
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# load MNIST dataset
test_datagen = ImageDataGenerator(rescale=1./255.,)

test_generator = test_datagen.flow_from_directory('./data/test', shuffle=False, batch_size=1, class_mode='categorical', target_size=(220, 220))

# server URL
# url = 'http://sports_ball_server:8501/v1/models/img_classifier:predict'
url = 'https://ca2-doaa03-ej-tf.herokuapp.com/v1/models/img_classifier:predict'


def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default",
                       "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions


def test_prediction(n_samples=3):
    for i, (X, y) in enumerate(test_generator):
        if i == n_samples:
            break

        prediction = make_prediction(X)
        assert np.argmax(y[0]) == np.argmax(prediction[0])