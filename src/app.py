from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('final_claim_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        region = int(request.form['region'])
        charges = float(request.form['charges'])

        # Create a DataFrame from the input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region],
            'charges': [charges]
        })

        # Make prediction
        prediction = model.predict(input_data)
        result = 'Denied' if prediction[0] == 1 else 'Approved'

        # Render the result
        return render_template('index.html', prediction_text=f'Predicted Claim Status: {result}')

    except Exception as e:
        return str(e)  # For debugging purposes

if __name__ == "__main__":
    app.run(debug=True)
