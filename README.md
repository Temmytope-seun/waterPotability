# Water Potability API

This project trains and serves a water potability prediction model using Flask.

## Overview

- **Training pipeline**: data ingestion, validation, transformation, model training, and evaluation
- **Serving layer**: Flask API with `/predict` for real-time inference
- **Configuration**: YAML-based config and environment variables

## Local Development

### 1. Prerequisites

- Python 3.10+
- pip
- Virtual environment (recommended)

### 2. Setup

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file (do not commit secrets):

```env
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

### 4. Run the Application

```bash
python app.py
```

The API will be available at:

- `http://localhost:5000/`
- `http://localhost:5000/predict`

## Training Workflow

Run the pipeline manually with:

```bash
python main.py
```

Or trigger the web route:

- `GET /train`

## API Usage

### Predict Endpoint

`POST /predict`

Expected form fields:

- `ph`
- `hardness`
- `solids`
- `chloramines`
- `sulfate`
- `conductivity`
- `organic_carbon`
- `trihalomethanes`
- `turbidity`

## Troubleshooting

- If the app cannot find the model, rerun the training pipeline.
- If validation fails, check the schema and input values.
- If the container cannot start, verify the `.env` file and mounted artifacts.

## Project Structure

- `app.py` — Flask web app
- `main.py` — training entry point
- `config/` — YAML configuration files
- `src/` — reusable pipeline and model logic
- `templates/` — HTML pages
- `tests/` — unit tests