from config import ROOT, PICTURE_PATH
from PIL import Image
import numpy as np
from dataclasses import dataclass


def load_picture(PATH, normalize=True):
    img = np.array(Image.open(PATH).convert('RGB'))

    if normalize:
        rgb_sum = img.sum(axis=1, keepdims=True)
        img = img / (rgb_sum + 1e-10)
        img = img[:, :, :2]

    return img

@dataclass
class Region:
    x1: int
    y1: int
    x2: int
    y2: int
    label: int

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def extract_features(img, regions):
    data = []
    labels = []

    for r in regions:
        features = img[r.y1 : r.y2, r.x1 : r.x2].reshape(-1, 2)
        data.append(features)

        label = np.full(len(features), r.label)
        labels.append(label)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)

    return data, labels
