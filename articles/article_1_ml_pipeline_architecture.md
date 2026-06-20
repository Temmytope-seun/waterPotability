# Building a Production-Ready ML Pipeline in Python: Architecture and Design Patterns

*How to structure a machine learning project that goes beyond the notebook*

---

## The Problem with Jupyter Notebooks in Production

Every data scientist knows the story. You build a beautiful model in a Jupyter notebook. The accuracy looks great. The business stakeholder is impressed. Then someone asks: "Can we deploy this?" And suddenly the notebook that took two weeks to build starts to feel like a house of cards.

The notebook has hardcoded file paths. The model parameters are buried in cell 17. There is no logging. There is no way to retrain without re-running every cell manually. The data download is a comment that says `# run this first`.

This is not a criticism of notebooks — they are excellent for exploration. But a notebook is not an application. Moving from a notebook to a reliable, maintainable ML system requires intentional design.

This article walks through a production-style ML pipeline for predicting red wine quality. The goal is not to build the most accurate model. The goal is to build a system that is readable, configurable, reproducible, and easy to extend.

---

## Project Structure at a Glance

The project is organised around a single source of truth: a `src/` package with four internal layers.

```
dscProject/
├── config/
│   └── config.yaml          # all file paths and URLs
├── params.yaml              # model hyperparameters
├── schema.yaml              # expected column names and types
├── src/dscProject/
│   ├── constants/           # file path constants
│   ├── entity/              # typed config dataclasses
│   ├── config/              # ConfigurationManager
│   ├── components/          # stage logic (data ingestion, training, etc.)
│   ├── pipeline/            # orchestration wrappers
│   └── utils/               # shared helpers
├── research/                # exploratory notebooks (isolated)
├── tests/                   # unit tests
├── app.py                   # Flask web application
└── main.py                  # full pipeline entry point
```

The key design decision is the strict separation between **what to do** (components) and **how to run it** (pipelines). Each layer has a single responsibility, and configuration flows from YAML files into typed Python objects before it reaches any business logic.

---

## The Five-Stage Pipeline

The pipeline runs five stages in sequence. Each stage is independent: it reads from an artifact produced by the previous stage and writes its own artifact to disk. This makes each stage individually testable and re-runnable without touching the others.

### Stage 1 — Data Ingestion

The first stage downloads a zip file from a remote URL, extracts it, and saves the raw CSV to the `artifacts/data_ingestion/` directory.

```python
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded")
        else:
            logger.info("File already exists, skipping download")

    def extract_zip_file(self):
        os.makedirs(self.config.unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
```

Notice the idempotency check: if the file already exists, the download is skipped. This is a small detail with a large practical impact — re-running the pipeline does not re-download 50 MB of data every time.

### Stage 2 — Data Validation

Before transforming anything, the pipeline checks that the incoming data has the expected columns. The expected schema is declared in `schema.yaml`:

```yaml
COLUMNS:
  fixed acidity: float64
  volatile acidity: float64
  citric acid: float64
  residual sugar: float64
  chlorides: float64
  free sulfur dioxide: float64
  total sulfur dioxide: float64
  density: float64
  pH: float64
  sulphates: float64
  alcohol: float64
  quality: int64
```

The validator compares the dataset columns against this schema and writes a status file:

```python
def validate_all_columns(self) -> bool:
    data = pd.read_csv(self.config.unzip_data_dir)
    all_cols = list(data.columns)
    all_schema = self.config.all_schema.keys()

    validation_status = all(col in all_schema for col in all_cols)

    with open(self.config.STATUS_FILE, "w") as f:
        f.write(f"Validation status: {validation_status}")
        if not validation_status:
            invalid_cols = set(all_cols) - set(all_schema)
            f.write(f"\nInvalid columns: {list(invalid_cols)}")

    return validation_status
```

If the upstream data source ever adds or renames a column, this stage catches it before the model tries to train on garbage.

### Stage 3 — Data Transformation

The transformation stage splits the validated data into training and test sets and saves them as separate CSV files. The split ratio could be added to `params.yaml` to make it configurable without touching any code.

```python
def train_test_split(self):
    df = pd.read_csv(self.config.data_path)
    train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)
    train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
    test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
```

In a more complex system, this stage would also handle feature engineering, scaling, and encoding. Keeping it as a dedicated stage means those operations can be added, changed, or removed without touching training or evaluation logic.

### Stage 4 — Model Training

Training uses scikit-learn's `ElasticNet` — a linear regression model with both L1 and L2 regularisation. ElasticNet is a solid baseline for tabular regression tasks. Its two hyperparameters, `alpha` and `l1_ratio`, are read from `params.yaml`:

```yaml
ElasticNet:
  alpha: 0.2
  l1_ratio: 0.1
```

The trainer loads training data, fits the model, and serialises it with `joblib`:

```python
def train_model(self):
    train_data = pd.read_csv(self.config.train_data_path)
    X_train = train_data.drop(columns=[self.config.target_column])
    y_train = train_data[self.config.target_column]

    model = ElasticNet(
        alpha=self.config.alpha,
        l1_ratio=self.config.l1_ratio,
        random_state=42
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(self.config.root_dir, "model", self.config.model_name)
    joblib.dump(model, model_path)
```

Changing the model from `ElasticNet` to a `RandomForestRegressor` only requires editing this file and `params.yaml`. Nothing else in the pipeline needs to change.

### Stage 5 — Model Evaluation

The evaluation stage loads the saved model, runs it on the held-out test set, and computes three regression metrics: RMSE, MAE, and R². These are saved to a JSON file and logged to MLflow for experiment tracking.

```python
def eval_metrics(self, actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    return rmse, mae, r2
```

---

## The Configuration System

The most important architectural decision in this project is keeping all configuration outside the code. Three YAML files drive the entire system:

- **`config.yaml`** — file paths, directory names, remote URLs
- **`params.yaml`** — model hyperparameters
- **`schema.yaml`** — expected data schema and target column name

These YAML files are read once at startup by `ConfigurationManager`, which assembles typed `@dataclass` config objects and passes them to each component:

```python
@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str
```

The `@dataclass` decorator gives you free `__init__`, `__repr__`, and type documentation. Combined with `@ensure_annotations` on the YAML reader, you get a lightweight type-safety layer without needing Pydantic.

This pattern means a data scientist can change the model's regularisation strength, the data source URL, or the target column name — all from YAML, without opening a single Python file.

---

## Why This Architecture Scales

This design is not over-engineered. It is the minimum structure needed for a real ML project to remain maintainable over time:

**Independent stages** mean you can rerun only the parts that changed. If you update the model hyperparameters, you do not need to re-download and re-validate the data.

**Config-driven** means environment-specific differences (local paths vs cloud paths, development data vs production data) can be handled with different YAML files rather than conditional logic scattered through the code.

**Typed config entities** mean the IDE can tell you what fields exist on a config object and catch typos before runtime.

**Separated components and pipelines** mean the same component logic can be orchestrated differently — by `main.py` on the command line, by Flask on a web request, or by Airflow in a scheduled job — without changing the component itself.

---

## Running the Full Pipeline

With all stages wired together, `main.py` runs the complete pipeline end-to-end:

```python
DataIngestionTrainingPipeline().initiate_data_ingestion()
DataValidationTrainingPipeline().initiate_data_validation()
DataTransformationTrainingPipeline().initiate_data_transformation()
ModelTrainingPipeline().initiate_model_trainer()
ModelEvaluationPipeline().initiate_model_evaluation()
```

Each pipeline class is a thin wrapper that wires the `ConfigurationManager` to its component and handles logging. The component does the work; the pipeline provides the plumbing.

---

## Key Takeaways

1. **Separate exploration from production.** Keep notebooks in `research/` for experimentation. The `src/` package is what ships.
2. **Drive configuration from YAML, not code.** Parameters that change between experiments belong in files, not function arguments.
3. **Type your config objects.** `@dataclass` is enough. It eliminates magic strings and makes your config self-documenting.
4. **Design stages to be idempotent.** A stage that can be safely re-run without side effects is a stage you can trust.
5. **Store artifacts explicitly.** Each stage should write its output to a named location. Future stages should read from that location, not from memory.

In the next article, we will look at the serving layer — how this pipeline connects to a Flask API and MLflow for experiment tracking, and what security practices matter when you move from a local machine to a deployed service.

---

*The full source code for this project is available on GitHub.*
