import cv2
import numpy as np


def predict_sign(model, image_path, image_size=30):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_id = np.argmax(prediction)
    return class_id
