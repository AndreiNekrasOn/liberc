import numpy as np
from time import sleep
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing import image

from dataset import convert_bitplane_to_image

np.set_printoptions(threshold=np.inf) # pyright: ignore


MODEL_NAME = ''
NAMED_PIPE_BITPLANE = '/tmp/dvc_pipe_bitplane'
NAMED_PIPE_LABEL = '/tmp/dvc_pipe_label'

# Note: order of labels is important

def load_model():
    return tf.keras.models.load_model(MODEL_NAME) # pyright: ignore

def to_bit_vector(zero_one_str: str):
    bits = [int(x) for x in zero_one_str.rstrip()]
    print(len(bits))
    return np.array(bits, dtype=np.uint8)

def generate_image(data):
    data = to_bit_vector(data)
    convert_bitplane_to_image(data, unpack=False)
    img = image.load_img("./out.png", target_size=(40,40))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    return img

def predict(model, data):
    img = generate_image(data)
    prediction = model.predict(img, verbose=0)
    prediction = prediction * 24 + 48
    prediction = max(prediction, 1584)
    prediction = min(prediction, 0)
    return prediction

def pipe_loop(model):
    print('Opening pipes...')
    bitplane_fd = os.open(NAMED_PIPE_BITPLANE, os.O_RDONLY)
    label_fd = os.open(NAMED_PIPE_LABEL, os.O_WRONLY)

    while True:
        print('Reading bitplane data...')
        data = os.read(bitplane_fd, 1585).decode('utf-8')
        if data == 'end':
            break
        if data == '':
            sleep(0.1)
            continue
        prediction = predict(model, data)
        print(f'Prediction: {prediction}')
        os.write(label_fd, "{:<5}".format(f'{prediction}\n').encode())
        print('Wrote label to pipe...')
    os.close(bitplane_fd)
    os.close(label_fd)


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='ML model name')
    args = parser.parse_args()
    MODEL_NAME = args.model
    print("Initializing ML model...")
    model = load_model()
    pipe_loop(model)
    
