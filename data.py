import numpy as np

def load_data(path):
    with np.load(path) as data:
        images = data['images']
        labels = data['labels']
    return images, labels

