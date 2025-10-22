from django.db import models

class PredictionHistory(models.Model):
    age = models.IntegerField()
    sex = models.IntegerField()
    cp = models.IntegerField()
    fbs = models.IntegerField()
    restecg = models.IntegerField()
    exang = models.IntegerField()
    slope = models.IntegerField()
    ca = models.IntegerField()
    thal = models.IntegerField()
    trestbps = models.IntegerField()
    chol = models.IntegerField()
    thalach = models.IntegerField()
    oldpeak = models.FloatField()
    prediction_result = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction on {self.created_at}"
