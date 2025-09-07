import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import uuid

OUTPUT_DIR = "gradcam_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_gradcam(model, mfcc, conv_layer_name, filename):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(mfcc)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-9)

    # Save heatmap
    heatmap_filename = f"{uuid.uuid4().hex}_{filename}.png"
    heatmap_path = os.path.join(OUTPUT_DIR, heatmap_filename)

    plt.matshow(heatmap)
    plt.axis("off")
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return heatmap_filename
