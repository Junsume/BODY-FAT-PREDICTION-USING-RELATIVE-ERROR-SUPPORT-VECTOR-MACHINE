from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained Random Forest model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        neck = float(request.form['neck'])
        chest = float(request.form['chest'])
        abdomen = float(request.form['abdomen'])
        hip = float(request.form['hip'])
        thigh = float(request.form['thigh'])
        knee = float(request.form['knee'])
        ankle = float(request.form['ankle'])
        biceps = float(request.form['biceps'])
        forearm = float(request.form['forearm'])
        wrist = float(request.form['wrist'])
        
        # Make prediction using the loaded model
        prediction = model.predict([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]])
        
        # Round the prediction to two decimal places
        predicted_body_fat = round(prediction[0], 2)
        
        # Return the prediction as JSON
        return jsonify({'predicted_body_fat': predicted_body_fat})

if __name__ == '__main__':
    app.run(debug=True)
