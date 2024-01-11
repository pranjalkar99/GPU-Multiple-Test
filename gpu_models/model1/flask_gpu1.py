from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load a pre-trained Inception ResNet V2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4"
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Preprocess image function
def preprocess_image(image, input_shape=(299, 299)):
    image = image.resize(input_shape)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# Route for prediction
@app.route('/predict')
def predict_resnet():
    # Generate a random image
    random_image = np.random.random((299, 299, 3)) * 255  # Random image with values between 0 and 255
    random_image = random_image.astype(np.uint8)  # Convert to uint8 (0-255)

    # Convert numpy array to PIL image
    image = Image.fromarray(random_image)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Perform inference
    predictions = model.predict(processed_image)

    

    return jsonify({'predictions': predictions.shape, 'gpu': len(tf.config.list_physical_devices('GPU')),'model':model_url})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5002)
