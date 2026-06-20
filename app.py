from flask import Flask, request, render_template
from dotenv import load_dotenv
import numpy as np

load_dotenv()
from src.waterPotability.pipeline.prediction_pipeline import PredictionPipeline
from src.waterPotability.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.waterPotability.pipeline.data_validation import DataValidationTrainingPipeline
from src.waterPotability.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.waterPotability.pipeline.model_trainer import ModelTrainingPipeline
from src.waterPotability.pipeline.model_evaluation import ModelEvaluationPipeline

app = Flask(__name__)

FEATURE_FIELDS = (
    'ph', 'hardness', 'solids', 'chloramines', 'sulfate',
    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity'
)


def build_feature_vector(form_data):
    return np.array(
        [float(form_data[field]) for field in FEATURE_FIELDS]
    ).reshape(1, len(FEATURE_FIELDS))


def get_prediction_label(result):
    return "Potable — Safe to Drink" if result == 1 else "Not Potable — Unsafe"


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
    if request.method != 'POST':
        return render_template('index.html')

    try:
        data = build_feature_vector(request.form)
        result = PredictionPipeline().predict(data)[0]
        return render_template(
            'results.html',
            prediction=get_prediction_label(result),
            safe=(result == 1)
        )
    except Exception as e:
        print('Exception:', str(e))
        return 'Something went wrong!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
