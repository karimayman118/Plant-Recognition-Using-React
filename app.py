from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import joblib
from tensorflow.keras.layers import Input
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)
CORS(app)
# Set the upload folder and allowed extensions
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the LabelEncoder
le = joblib.load('label_encoder.pkl')  # Load pre-saved LabelEncoder
num_classes = len(le.classes_)
print(f"LabelEncoder loaded with classes: {le.classes_}")

# Define model architecture
ScaleTo = 70  # px to scale



model = Sequential()

model.add(Input(shape=(ScaleTo, ScaleTo, 3))) 

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# Load pre-trained weights
model.load_weights('weights.last_auto4.hdf5')



# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image
def preprocess_image(img_path, scale_to, lower_green, upper_green):
    # Read and resize image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (scale_to, scale_to))

    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to HSV and create mask
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to the image
    b_mask = mask > 0
    clear_img = np.zeros_like(img, np.uint8)
    clear_img[b_mask] = img[b_mask]

    # Normalize image
    clear_img = clear_img / 255.0
    return clear_img

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Define the green color range for masking
        lower_green = (25, 40, 50)
        upper_green = (75, 255, 255)

        # Preprocess the uploaded image
        processed_image = preprocess_image(filepath, ScaleTo, lower_green, upper_green)

        # Expand dimensions to match the model's input shape
        processed_image = np.expand_dims(processed_image, axis=0)

        # Predict using the model
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction, axis=1)[0]

        # Map the predicted label to a class name (use LabelEncoder or predefined list)
        class_names = le.classes_  # If using LabelEncoder, or define your own list of plant names
        result = class_names[predicted_label] if predicted_label < len(class_names) else 'Unknown'

        # Convert image to base64 to send back to frontend
        img = Image.open(filepath)  # Open the image using PIL
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the prediction and the base64-encoded image
        return jsonify({
            'prediction': result,
            'image': img_base64
        })

    return jsonify({'error': 'Invalid file format'})


    

# Run the app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
