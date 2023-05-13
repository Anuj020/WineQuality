from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application = Flask(__name__)
app = application

## Route for a home page

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['get','post'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            fixed_acidity = request.form.get('fixed_acidity'),
            volatile_acidity = request.form.get('volatile_acidity'),
            citric_acid = request.form.get('citric_acid'),
            residual_sugar = request.form.get('residual_sugar'),
            chlorides = request.form.get('chlorides'),
            free_sulfur_dioxide = request.form.get('free_sulfur_dioxide'),
            total_sulfur_dioxide = request.form.get('total_sulfur_dioxide'),
            density = request.form.get('density'),
            pH = request.form.get('pH'),
            sulphates = request.form.get('sulphates'),
            alcohol = request.form.get('alcohol'),
          
          )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results = results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)