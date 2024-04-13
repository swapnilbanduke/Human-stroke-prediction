from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('model/model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Print received form data
        print("Received form data:", request.form)

        # Get the data from the form
        gender = int(request.form.get('gender')) if request.form.get('gender') is not None else None
        age = int(request.form.get('age')) if request.form.get('age') is not None else None
        hypertension = int(request.form.get('hypertension')) if request.form.get('hypertension') is not None else None
        heart_disease = int(request.form.get('heart_disease')) if request.form.get('heart_disease') is not None else None
        ever_married = int(request.form.get('ever_married')) if request.form.get('ever_married') is not None else None
        avg_glucose_level = float(request.form.get('avg_glucose_level')) if request.form.get('avg_glucose_level') is not None else None
        bmi = float(request.form.get('bmi')) if request.form.get('bmi') is not None else None
        work_type_Govt_job = int(request.form.get('work_type_Govt_job')) if request.form.get('work_type_Govt_job') is not None else None
        work_type_Never_worked = int(request.form.get('work_type_Never_worked')) if request.form.get('work_type_Never_worked') is not None else None
        work_type_Private = int(request.form.get('work_type_Private')) if request.form.get('work_type_Private') is not None else None
        work_type_Self_employed = int(request.form.get('work_type_Self_employed')) if request.form.get('work_type_Self_employed') is not None else None
        work_type_children = int(request.form.get('work_type_children')) if request.form.get('work_type_children') is not None else None
        residence_type_Rural = int(request.form.get('residence_type_Rural')) if request.form.get('residence_type_Rural') is not None else None
        residence_type_Urban = int(request.form.get('residence_type_Urban')) if request.form.get('residence_type_Urban') is not None else None
        smoking_status_Unknown = int(request.form.get('smoking_status_Unknown')) if request.form.get('smoking_status_Unknown') is not None else None
        smoking_status_formerly_smoked = int(request.form.get('smoking_status_formerly_smoked')) if request.form.get('smoking_status_formerly_smoked') is not None else None
        smoking_status_never_smoked = int(request.form.get('smoking_status_never_smoked')) if request.form.get('smoking_status_never_smoked') is not None else None
        smoking_status_smokes = int(request.form.get('smoking_status_smokes')) if request.form.get('smoking_status_smokes') is not None else None

        # Check if any required field is None
        if None in [gender, age, hypertension, heart_disease, ever_married,
                    avg_glucose_level, bmi, work_type_Govt_job, work_type_Never_worked,
                    work_type_Private, work_type_Self_employed, work_type_children,
                    residence_type_Rural, residence_type_Urban, smoking_status_Unknown,
                    smoking_status_formerly_smoked, smoking_status_never_smoked,
                    smoking_status_smokes]:
            # Print missing fields
            print("Some form fields are missing:", {
                'gender': gender, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
                'ever_married': ever_married, 'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
                'work_type_Govt_job': work_type_Govt_job, 'work_type_Never_worked': work_type_Never_worked,
                'work_type_Private': work_type_Private, 'work_type_Self_employed': work_type_Self_employed,
                'work_type_children': work_type_children, 'residence_type_Rural': residence_type_Rural,
                'residence_type_Urban': residence_type_Urban, 'smoking_status_Unknown': smoking_status_Unknown,
                'smoking_status_formerly_smoked': smoking_status_formerly_smoked,
                'smoking_status_never_smoked': smoking_status_never_smoked, 'smoking_status_smokes': smoking_status_smokes
            })
            return "Some form fields are missing", 400

        # Create a DataFrame with the new data
        new_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married,
                                   avg_glucose_level, bmi, work_type_Govt_job, work_type_Never_worked,
                                   work_type_Private, work_type_Self_employed, work_type_children,
                                   residence_type_Rural, residence_type_Urban, smoking_status_Unknown,
                                   smoking_status_formerly_smoked, smoking_status_never_smoked,
                                   smoking_status_smokes]])

        # Make prediction
        prediction = model.predict(new_data)

        # Determine the result
        if prediction[0] == 1:
            result = 'Person Have Stroke'
        else:
            result = 'Person is Healthy'

        return render_template('single_prediction.html', prediction_result=result)
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


        