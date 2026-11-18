import numpy as np
import matplotlib.pyplot as plt
from classifiers import (
    minimum_error, minimum_error_predict, error_rate
)
from utils import (
    load_picture, extract_features,
    Dataset, Region
)

from PIL import Image
from config import ROOT, PICTURE_PATH

# ---------------------------------------------------------------

if __name__ == '__main__':

    TRAIN_PATH = PICTURE_PATH / 'Bilde2.png'
    TEST_PATH = PICTURE_PATH / 'Bilde3.png'

    train_img = load_picture(TRAIN_PATH)
    test_img = load_picture(TEST_PATH)
    
    regions = [
        Region(2000, 400, 2100, 500, 0),  # Region for red folder
        Region(700, 900, 800, 1000, 1),  # Region for blue folder
        Region(10, 10, 60, 60, 2)     # Region for background
    ]

    X_train, y_train = extract_features(train_img, regions)
    X_out = train_img.reshape(-1, 2)
    X_test = test_img.reshape(-1, 2)

    params = minimum_error(X_train, y_train)
    test_result = minimum_error_predict(params, X_test).reshape(test_img.shape[:2])
    train_result = minimum_error_predict(params, X_out).reshape(train_img.shape[:2])

    # result_img = Image.fromarray()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(train_result, cmap='viridis')
    ax2.imshow(test_result, cmap='viridis')

    plt.show()

    