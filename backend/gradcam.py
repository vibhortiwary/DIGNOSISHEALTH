# backend/gradcam.py
import tensorflow as tf
import numpy as np
import cv2, os

IMG_SIZE = (224, 224)

def generate_gradcam(model, img_path, class_index):
    """
    Fixed Grad-CAM for MobileNetV3.
    """

    # Load img
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)

    preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(
        img_arr.copy()
    )
    input_tensor = np.expand_dims(preprocessed, axis=0)

    # Backbone inside your model
    base = model.get_layer("MobilenetV3small")

    # Last conv
    last_conv_layer = None
    for layer in reversed(base.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D in backbone")

    # Rebuild classifier head
    gap = base.output
    x = model.layers[-3](gap)
    x = model.layers[-2](x)
    preds = model.layers[-1](x)

    grad_model = tf.keras.models.Model(
        inputs=base.input,
        outputs=[last_conv_layer.output, preds],
    )

    # Gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, IMG_SIZE)

    cam -= cam.min()
    cam /= cam.max() + 1e-10

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    original = cv2.resize(img_arr.astype(np.uint8), IMG_SIZE)
    overlay = cv2.addWeighted(original, 0.5, heatmap, 0.5, 0)

    os.makedirs("backend/gradcam", exist_ok=True)
    out_path = f"backend/gradcam/gradcam_{os.path.basename(img_path)}"
    cv2.imwrite(out_path, overlay)

    return out_path
