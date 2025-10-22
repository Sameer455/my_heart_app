<<<<<<< HEAD
# 🩺 Heart Disease Prediction Web Application

A comprehensive web application that uses machine learning algorithms to predict heart disease risk based on medical parameters. Built with Django and featuring a modern, responsive UI.

## 🌟 Features

- **Real-time Heart Disease Prediction** - Instant ML-based risk assessment
- **Multiple ML Algorithms** - 6 different algorithms with automatic best model selection
- **Prediction History** - Complete audit trail of all predictions
- **Health Recommendations** - Diet plans and workout routines for heart health
- **Modern UI/UX** - Glassmorphism design with responsive layout
- **Data Visualization** - Comprehensive EDA charts and analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Django application**
   ```bash
   cd heart_disease_project
   python manage.py migrate
   python manage.py runserver
   ```

4. **Access the application**
   - Open your browser and go to `http://localhost:8000/`

## 🧠 Machine Learning

### Algorithms Used
- **Logistic Regression** - Linear classification baseline
- **Decision Tree** - Interpretable model with feature importance
- **Random Forest** - Ensemble method (Best: 88.52% accuracy)
- **Support Vector Classifier** - Non-linear classification
- **K-Nearest Neighbors** - Instance-based learning
- **XGBoost** - Gradient boosting for high performance

### Model Performance
- **Accuracy**: 88.52%
- **Cross-Validation**: 3-fold CV
- **Best Algorithm**: Random Forest Classifier
- **Features**: 13 input features (8 categorical + 5 numerical)

## 🏗️ Technical Stack

### Backend
- **Django 5.0.7** - Web framework
- **Python 3.12** - Programming language
- **SQLite/PostgreSQL** - Database
- **Django ORM** - Database abstraction

### Machine Learning
- **Scikit-learn 1.3.2** - ML algorithms
- **Pandas 2.1.3** - Data manipulation
- **NumPy 1.26.4** - Numerical computing
- **XGBoost 1.7.6** - Gradient boosting
- **Joblib 1.5.2** - Model serialization

### Frontend
- **Bootstrap 5.3.0** - CSS framework
- **jQuery 3.6.0** - JavaScript library
- **Custom CSS** - Glassmorphism design
- **HTML5** - Responsive markup

## 📊 Dataset Information

- **Source**: Heart Disease Dataset (heart.csv)
- **Records**: 302 (after duplicate removal)
- **Features**: 13 input features + 1 target variable
- **Target**: Binary classification (0: No Heart Disease, 1: Heart Disease)

### Input Features
1. **age** - Age in years (29-77)
2. **sex** - Gender (0: Female, 1: Male)
3. **cp** - Chest Pain Type (0-3)
4. **trestbps** - Resting Blood Pressure (94-200 mm Hg)
5. **chol** - Serum Cholesterol (126-564 mg/dl)
6. **fbs** - Fasting Blood Sugar > 120 mg/dl (0/1)
7. **restecg** - Resting ECG Results (0-2)
8. **thalach** - Maximum Heart Rate (71-202 bpm)
9. **exang** - Exercise Induced Angina (0/1)
10. **oldpeak** - ST Depression (0.0-6.2)
11. **slope** - ST Segment Slope (0-2)
12. **ca** - Major Vessels (0-3)
13. **thal** - Thalassemia (0-3)

## 🎯 Usage

### Making Predictions
1. Fill in the medical parameters in the prediction form
2. Click "Predict" to get instant heart disease risk assessment
3. View results with clear indicators (✅ No Heart Disease / ⚠️ Heart Disease Detected)

### Additional Features
- **About Page** - Application information and disclaimers
- **Diet Plans** - Heart-healthy meal recommendations
- **Workout Plans** - Exercise routines for different fitness levels
- **History** - View all previous predictions with timestamps

## 🗄️ Database

### Current: SQLite
- File-based database for easy development
- Automatic migrations included

### PostgreSQL Migration
- Migration script provided (`postgresql_setup.py`)
- Production-ready configuration available
- Update `settings.py` for PostgreSQL connection

## 📁 Project Structure

```
heart-disease-prediction/
├── heart_disease_project/          # Django application
│   ├── manage.py
│   ├── heart_disease_project/
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   └── predictor/                  # Main app
│       ├── models.py
│       ├── views.py
│       ├── urls.py
│       └── templates/
│           └── predictor/
│               ├── index.html      # Prediction form
│               ├── about.html      # App information
│               ├── diet.html       # Diet recommendations
│               ├── workout.html    # Exercise plans
│               └── history.html    # Prediction history
├── heart.csv                       # Dataset
├── model_build.py                  # ML training script
├── requirements.txt                # Dependencies
├── PROJECT_REPORT.md              # Comprehensive technical report
├── TECHNICAL_SUMMARY.md           # Quick reference guide
└── README.md                      # This file
```

## 🔧 Development

### Model Retraining
```bash
python model_build.py
```

### Quick Model Retraining (Compatibility)
```bash
python quick_model_retrain.py
```

### Database Migrations
```bash
cd heart_disease_project
python manage.py makemigrations
python manage.py migrate
```

## 📈 Performance

- **Prediction Speed**: < 1 second
- **Model Accuracy**: 88.52%
- **Response Time**: Real-time AJAX predictions
- **Scalability**: Ready for production deployment

## 🛡️ Security

- **CSRF Protection** - Django's built-in CSRF middleware
- **Input Validation** - HTML5 and server-side validation
- **Error Handling** - Comprehensive exception management
- **Data Privacy** - Local processing, no external API calls

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

This application is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.


## 🎉 Acknowledgments

- Heart Disease Dataset providers
- Django community for the excellent web framework
- Scikit-learn team for machine learning tools
- Bootstrap team for the responsive UI framework

---

**Made with ❤️ for Heart Health Awareness**

*Last Updated: October 2025*
=======
# my_heart_app
>>>>>>> 01d5742cb01e9830be0ea016101a45a090ae2651
