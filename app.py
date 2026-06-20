from flask import Flask, request, render_template
import numpy as np
from src.dscProject.pipeline.prediction_pipeline import PredictionPipeline
from src.dscProject.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.dscProject.pipeline.data_validation import DataValidationTrainingPipeline
from src.dscProject.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.dscProject.pipeline.model_trainer import ModelTrainingPipeline
from src.dscProject.pipeline.model_evaluation import ModelEvaluationPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    DataIngestionTrainingPipeline().initiate_data_ingestion()
    DataValidationTrainingPipeline().initiate_data_validation()
    DataTransformationTrainingPipeline().initiate_data_transformation()
    ModelTrainingPipeline().initiate_model_trainer()
    ModelEvaluationPipeline().initiate_model_evaluation()
    return render_template('train.html')

@app.route('/predict', methods=['POST', 'GET'])
def evaluate():
    if request.method == 'POST':
        try:
            ph                  = float(request.form['ph'])
            hardness            = float(request.form['hardness'])
            solids              = float(request.form['solids'])
            chloramines         = float(request.form['chloramines'])
            sulfate             = float(request.form['sulfate'])
            conductivity        = float(request.form['conductivity'])
            organic_carbon      = float(request.form['organic_carbon'])
            trihalomethanes     = float(request.form['trihalomethanes'])
            turbidity           = float(request.form['turbidity'])

            data = np.array([
                ph, hardness, solids, chloramines, sulfate,
                conductivity, organic_carbon, trihalomethanes, turbidity
            ]).reshape(1, 9)

            result = PredictionPipeline().predict(data)[0]
            label = "Potable — Safe to Drink" if result == 1 else "Not Potable — Unsafe"
            return render_template('results.html', prediction=label, safe=(result == 1))
        except Exception as e:
            print('Exception:', str(e))
            return 'Something went wrong!'
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
