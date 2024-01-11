import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Check if GPU is available and visible to TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU is available")
else:
    print("No GPU detected")

# Load a pre-trained Inception ResNet V2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Test inference with random data
input_shape = (299, 299, 3)  # Inception ResNet V2 input shape
num_samples = 5
random_images = np.random.random((num_samples, *input_shape))

# Normalize the random data to match the Inception ResNet V2 input range
preprocessed_images = tf.keras.applications.inception_resnet_v2.preprocess_input(random_images)

# Perform inference on GPU
with tf.device('/GPU:0'):
    predictions = model.predict(preprocessed_images)

print(predictions)