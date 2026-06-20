# MLOps in Practice: Experiment Tracking, Model Serving, and Security

*From a trained model file to a live API — and the pitfalls in between*

---

## Where Most ML Tutorials Stop

Most machine learning tutorials end at `model.fit()`. The accuracy is printed. The tutorial is done. But in a real project, training the model is not the end — it is the beginning of a different kind of work.

Once a model is trained you need to answer three questions:

1. **How do I know if this run was better than the last one?** (Experiment tracking)
2. **How do other systems use this model?** (Serving)
3. **How do I make sure this is safe to run in production?** (Security)

This article covers all three, using a wine quality prediction project as a concrete example.

---

## Part 1: Experiment Tracking with MLflow and DagsHub

### The Problem MLflow Solves

Without experiment tracking, a typical ML workflow looks like this:

```
# train_v1.py  →  accuracy: 0.72
# train_v2.py  →  accuracy: 0.74  (what did I change?)
# train_v3_FINAL.py  →  accuracy: 0.71  (why did it go down?)
# train_v3_FINAL_use_this_one.py  →  ...
```

MLflow replaces this chaos with a structured record of every run: which parameters were used, what metrics were produced, and which model artefact was saved.

### How It Is Wired Up

In the `ModelEvaluation` component, after computing metrics on the held-out test set, the results are logged to a remote MLflow tracking server hosted on DagsHub:

```python
def log_into_mlflow(self):
    test_data = pd.read_csv(self.config.test_data_path)
    model = joblib.load(self.config.model_path)

    X_test = test_data.drop([self.config.target_column], axis=1)
    y_test = test_data[self.config.target_column]

    mlflow.set_tracking_uri(self.config.mlflow_uri)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    with mlflow.start_run():
        y_pred = model.predict(X_test)
        rmse, mae, r2 = self.eval_metrics(y_test, y_pred)

        # Persist metrics locally as JSON
        save_json(path=Path(self.config.metrics_file_path), data={
            "RMSE": rmse, "MAE": mae, "R2_Score": r2
        })

        # Log parameters and metrics to MLflow
        mlflow.log_params(self.config.all_params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_Score", r2)

        # Register the model if using a remote store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model",
                                     registered_model_name="ElasticNetModel")
        else:
            mlflow.sklearn.log_model(model, "model")
```

Three things are happening here:

**Metrics are saved twice.** They go to a local JSON file (`metrics.json`) and to the remote MLflow server. The local copy is useful for quick inspection without an internet connection. The remote copy is what you actually compare across runs.

**Parameters are logged alongside metrics.** When you look back at a run three months later, you will see that `alpha=0.2` and `l1_ratio=0.1` produced `RMSE=0.58`. Without logging parameters, the metric number is meaningless.

**The model registration step is conditional.** MLflow's model registry only works with a remote tracking store (like DagsHub), not with a local file-based store. The `urlparse` check handles both cases cleanly.

### What You See in DagsHub

After a run, the DagsHub MLflow UI shows a table of experiments. Each row is one run, with columns for every logged parameter and metric. You can sort by `R2_Score`, compare runs side by side, and see the exact model artefact that produced each result.

This is the foundation of reproducible ML. If a stakeholder asks "which model is in production and how good was it?", you have a precise, auditable answer.

---

## Part 2: Serving the Model with Flask

### The Prediction Pipeline

At prediction time, the serialised `model.joblib` file is loaded once when the `PredictionPipeline` is instantiated:

```python
class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model/model.joblib'))

    def predict(self, data):
        return self.model.predict(data)
```

Loading the model in `__init__` rather than inside `predict` is important for performance. If you load the model on every request, a busy API wastes the majority of its time on disk I/O.

### The Flask Application

The Flask app exposes three routes:

```python
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
        fixed_acidity        = float(request.form['fixed_acidity'])
        volatile_acidity     = float(request.form['volatile_acidity'])
        citric_acid          = float(request.form['citric_acid'])
        residual_sugar       = float(request.form['residual_sugar'])
        chlorides            = float(request.form['chlorides'])
        free_sulfur_dioxide  = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density              = float(request.form['density'])
        pH                   = float(request.form['pH'])
        sulphates            = float(request.form['sulphates'])
        alcohol              = float(request.form['alcohol'])

        data = np.array([
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        ]).reshape(1, 11)

        prediction = PredictionPipeline().predict(data)
        return render_template('results.html', prediction=str(prediction))
    return render_template('index.html')
```

**`/`** — Serves a form where a user can enter the 11 physicochemical features of a wine sample.

**`/train`** — Triggers the full pipeline: ingest → validate → transform → train → evaluate. This is a blocking call, so the browser will wait until all five stages complete. For a production system you would want to run this as a background task and poll for completion, but for a learning project the synchronous approach is transparent and easy to reason about.

**`/predict`** — Reads 11 float values from the submitted form, assembles them into a numpy array, runs the model, and renders the prediction on a results page.

### Input Validation

The `float()` calls on each form field provide basic type coercion, but they will raise an unhandled `ValueError` if the user submits a non-numeric value. A more robust implementation would validate bounds (alcohol cannot be negative; pH cannot exceed 14) and return a meaningful error message rather than a 500 page. This is the next natural improvement for this route handler.

---

## Part 3: Security — The Mistakes That Commonly Get Made

This section covers two security issues that this project originally had, and how they were fixed. Both are extremely common in ML projects.

### Mistake 1: Hardcoded Credentials in Source Code

The original `model_evaluation.py` contained this:

```python
os.environ["MLFLOW_TRACKING_USERNAME"] = "Temmytope-seun"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "ffd4eb226da5bd1070fe9492741f8ef8728c0623"
```

This is a credential leak. The moment this file is committed to a git repository — even a private one — the secret is in the git history permanently. Anyone with read access to the repository can recover it with `git log`.

The correct approach is to never write credentials in code. Instead:

1. Set them as environment variables before the process starts.
2. Document which variables are required in a `.env.example` file.
3. Add `.env` to `.gitignore` so the actual values are never committed.

```bash
# .env.example  (committed — shows structure, no real values)
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_access_token
```

MLflow reads `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` automatically from the environment when it connects to a remote tracking server. You do not need to set them in code at all.

If you have already committed credentials, the immediate actions are:
- Rotate the token on the provider's website (DagsHub → Settings → Access Tokens → Revoke)
- Remove the hardcoded values from the code
- Note that the old token remains in git history; tools like `git-filter-repo` can rewrite history if necessary, though this is disruptive on shared branches

### Mistake 2: Shell Injection via `os.system`

The original training route used:

```python
os.system('python main.py')
```

`os.system` passes a string to the shell. In this specific case the string is a literal, so there is no immediate injection risk. But `os.system` is the wrong tool for the job for two reasons:

**It is opaque.** `os.system` returns an exit code, not the output. You cannot programmatically check whether the training succeeded, capture error messages, or surface them to the user.

**It spawns a shell.** A shell is an unnecessary intermediary between your Python process and another Python process. Any future refactor that adds user-controlled content to that string creates a shell injection vulnerability.

The correct fix is a direct Python function call:

```python
@app.route('/train', methods=['GET'])
def train():
    DataIngestionTrainingPipeline().initiate_data_ingestion()
    DataValidationTrainingPipeline().initiate_data_validation()
    DataTransformationTrainingPipeline().initiate_data_transformation()
    ModelTrainingPipeline().initiate_model_trainer()
    ModelEvaluationPipeline().initiate_model_evaluation()
    return render_template('train.html')
```

No shell. No subprocess. Exceptions propagate naturally. The Flask error handler can catch them and return a proper HTTP error response.

---

## Containerising with Docker

The project includes a `Dockerfile` for containerised deployment. This solves the "it works on my machine" problem by packaging the application and all its dependencies into an image that runs identically everywhere.

A typical workflow:

```bash
# Build
docker build -t wine-quality-api .

# Run locally
docker run -p 5000:5000 \
  -e MLFLOW_TRACKING_USERNAME=your_username \
  -e MLFLOW_TRACKING_PASSWORD=your_token \
  wine-quality-api
```

Notice that credentials are passed as environment variables at runtime, not baked into the image. This is the standard pattern for containerised secrets.

---

## The Full MLOps Loop

Putting it all together, the lifecycle of this system looks like this:

```
1. User hits /train
        ↓
2. Data is downloaded and validated
        ↓
3. Data is split into train/test
        ↓
4. ElasticNet model is trained with params from params.yaml
        ↓
5. Model is evaluated on test set
        ↓
6. Metrics (RMSE, MAE, R²) are logged to MLflow on DagsHub
        ↓
7. Model is registered in the MLflow Model Registry
        ↓
8. User hits /predict with wine features
        ↓
9. Saved model returns a quality prediction
```

Every step in this loop is logged, every artefact is versioned, and every parameter is recorded. If the model degrades, you can trace back through the experiment history to understand why.

---

## Key Takeaways

1. **Log parameters alongside metrics.** A metric without the parameters that produced it is not reproducible.
2. **Load the model once, not per request.** Put expensive I/O in `__init__`, not in the hot path.
3. **Never hardcode credentials.** Use environment variables. Rotate any token that was ever committed to source control.
4. **Avoid `os.system` for calling Python code.** Call Python functions directly. Reserve subprocess for external tools.
5. **Pass secrets as runtime environment variables to Docker**, not as build arguments or image layers.

---

*Part 1 of this series covered the pipeline architecture. Part 3 covers how to write reliable unit tests for every stage of the pipeline.*
