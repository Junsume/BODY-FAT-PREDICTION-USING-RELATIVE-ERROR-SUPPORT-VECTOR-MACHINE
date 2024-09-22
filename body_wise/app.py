from flask import Flask, render_template, request, jsonify
from final_svr import IRE_SVM  # Import your IRE_SVM class
import numpy as np

app = Flask(__name__)

# Load the trained model
ire_svm = IRE_SVM(C=4*10**7, gamma_b=1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form data
        features = np.array([[float(request.form[field]) for field in ['age', 'weight', 'height', 'neck', 'chest', 'abdomen', 'hip', 'thigh', 'knee', 'ankle', 'biceps', 'forearm', 'wrist']]]).reshape(1, -1)
        
        # Make prediction using the pre-initialized model instance
        prediction = ire_svm.predict(features)
        
        # Return prediction as JSON
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        # If an error occurs, return error message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
