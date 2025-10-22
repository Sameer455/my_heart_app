import os
import joblib
import dill
import pandas as pd
import numpy as np
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import PredictionHistory

# Load the trained model and encoders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "predictor/overall_best_model.pkl"))

with open(os.path.join(BASE_DIR, "predictor/ohe_encoder.dill"), "rb") as f:
    ohe_encoder = dill.load(f)

with open(os.path.join(BASE_DIR, "predictor/standard_encoder.dill"), "rb") as f:
    scaler = dill.load(f)

# Categorical and numerical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Render Pages
def index(request):
    return render(request, 'predictor/index.html')

def about(request):
    return render(request, 'predictor/about.html')

def diet(request):
    return render(request, 'predictor/diet.html')

def workout(request):
    return render(request, 'predictor/workout.html')

def history(request):
    """Display past predictions."""
    history_data = PredictionHistory.objects.all().order_by('-created_at')
    return render(request, 'predictor/history.html', {'history': history_data})

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Convert data to DataFrame
            input_data = pd.DataFrame([data])

            # One-Hot Encoding for categorical columns
            input_data_encoded = ohe_encoder.transform(input_data[categorical_columns])

            # Scaling numerical columns
            input_data_scaled = scaler.transform(input_data[numerical_columns])

            # Combine encoded and scaled features
            input_data_final = np.hstack((input_data_scaled, input_data_encoded))

            # Make a prediction
            prediction = model.predict(input_data_final)[0]

            # Store in database
            try:
                PredictionHistory.objects.create(
                    age=data['age'], sex=data['sex'], cp=data['cp'],
                    fbs=data['fbs'], restecg=data['restecg'], exang=data['exang'],
                    slope=data['slope'], ca=data['ca'], thal=data['thal'],
                    trestbps=data['trestbps'], chol=data['chol'],
                    thalach=data['thalach'], oldpeak=data['oldpeak'],
                    prediction_result="Heart Disease Detected" if prediction == 1 else "No Heart Disease"
                )
            except Exception as db_error:
                print(f"Database error: {db_error}")
                # Continue with prediction even if database save fails

            return JsonResponse({'prediction': int(prediction)})
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)
