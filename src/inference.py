import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys

def predict_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    
    pred = model.predict(x)
    class_label = "Dog" if pred[0][0] > 0.5 else "Cat"
    print(f"Prediction: {class_label} ({pred[0][0]:.4f})")
    return class_label

if __name__ == "__main__":
    if len(sys.argv) > 2:
        predict_image(sys.argv[1], sys.argv[2])
