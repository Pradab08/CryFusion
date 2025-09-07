import tensorflow as tf
from tensorflow import keras
import os

# Paths
MODEL_PATH = "models/cnn_lstm.keras"
FROZEN_KERAS_PATH = "models/cnn_lstm_frozen.keras"
SAVEDMODEL_DIR = "models/cnn_lstm_savedmodel"

print("🔹 Loading trained model:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

# Freeze model
print("🔹 Freezing layers...")
for layer in model.layers:
    layer.trainable = False

# Re-wrap into a new Model to detach training artifacts
inputs = keras.Input(shape=model.input_shape[1:])
outputs = model(inputs, training=False)
frozen_model = keras.Model(inputs, outputs, name="cnn_lstm_frozen")

# Save frozen Keras model
print("🔹 Saving frozen Keras model:", FROZEN_KERAS_PATH)
frozen_model.save(FROZEN_KERAS_PATH)

# Save as TensorFlow SavedModel
print("🔹 Exporting SavedModel:", SAVEDMODEL_DIR)
# clear any compilation state (important!)
frozen_model.compile()
tf.saved_model.save(frozen_model, SAVEDMODEL_DIR)

print("✅ Model freezing complete!")