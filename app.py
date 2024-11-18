from flask import Flask, render_template, request
import os
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Define the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}

# Use raw string to avoid invalid escape sequence warning
model_path = r'D:\COVID_DETECTION\App\model\VGG.h5'  # Update the path to your model with the correct extension
model = load_model(model_path)  # Load the Keras model

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message="No file selected")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected")

    if file and allowed_file(file.filename):
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Prepare the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))  # Update the target size based on your model input
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the model
        result = model.predict(img_array)

        # Assuming the model outputs categories or probabilities, handle the result accordingly
        predicted_category = 'Positive' if result[0][0] > 0.5 else 'Negative'  # Modify based on model output
        
        # Pass only the filename to the result page
        return render_template('result.html', result=predicted_category, image_path=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
