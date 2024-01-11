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

# Load a pre-trained MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Test inference with random data
input_shape = (224, 224, 3)
num_samples = 5
random_images = np.random.random((num_samples, *input_shape))

# Normalize the random data to match the MobileNetV2 input range
preprocessed_images = tf.keras.applications.mobilenet_v2.preprocess_input(random_images)

# Perform inference on GPU
with tf.device('/GPU:0'):
    predictions = model.predict(preprocessed_images)

# Decode predictions (if needed)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

# Display the predictions
for i in range(num_samples):
    print(f"Predictions for image {i+1}:")
    for prediction in decoded_predictions[i]:
        print(f"{prediction[1]}: {prediction[2]:.4f}")
    print()
