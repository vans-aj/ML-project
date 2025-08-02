from flask import Flask,request,render_template 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # this is for scaling the data 
from src.pipelines.predict_pipeline import predictpipeline, CustomData

application = Flask(__name__)

app = application

#route for home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST', 'GET'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else :
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch= request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            reading_score= int(request.form.get('reading_score')),
            writing_score= int(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        pred_pipline = predictpipeline()
        results = pred_pipline.predict(features=pred_df)
        return render_template('home.html',results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)



