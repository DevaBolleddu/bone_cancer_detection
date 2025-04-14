# explainability.py
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np

def grad_cam(model, img_array, layer_name):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]
    grads = tape.gradient(loss, conv_outputs)[0]
    cam = np.mean(grads, axis=(0, 1))
    return cam

def shap_explain(model, data):
    explainer = shap.Explainer(model.predict, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data)
