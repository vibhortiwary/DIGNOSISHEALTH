# backend/brain_loader.py

import tensorflow as tf
import json

IMG_SIZE = (224, 224)

def load_brain_model():
    """
    Load the full MobileNetV3 brain model saved as .keras (SavedModel format)
    + class map.
    """

    # Load class map
    class_map = json.load(open("backend/models/brain_class_map_best.json"))

    # Load FULL model (architecture + weights)
    model = tf.keras.models.load_model("backend/models/cnn_brain_best.keras")

    return model, class_map
