# Unit Testing Machine Learning Pipelines: A Practical Guide

*Why testing ML code is hard, and exactly how to do it anyway*

---

## The Myth That ML Code Cannot Be Tested

Ask most data scientists whether their pipeline has unit tests and the answer is often some variation of: "It's machine learning — you can't really unit test it. The output is probabilistic."

This is a misunderstanding of what unit tests are for. Unit tests do not check whether the model is accurate. They check whether the code does what it is supposed to do. And ML pipeline code — downloading files, validating schemas, splitting data, saving models, logging metrics — is deterministic code that absolutely can and should be tested.

This article walks through 20 unit tests written for a five-stage ML pipeline. By the end, you will understand the three main techniques for testing ML code and have concrete patterns you can apply to your own projects.

---

## Why ML Code Is Harder to Test Than Web Code

There are three specific challenges that make ML pipelines trickier to test than a typical web backend:

**1. Heavy I/O.** Almost every function reads from or writes to disk. A unit test that actually reads a 1,000-row CSV or writes a model file is slow, fragile, and leaves artefacts behind.

**2. External dependencies.** Stages call `mlflow.log_metric`, `urllib.request.urlretrieve`, or `zipfile.ZipFile`. A test that actually hits a remote MLflow server is not a unit test — it is an integration test that requires network access and valid credentials.

**3. Non-determinism.** Model training involves random number generators. The exact weights learned will differ between runs unless you control `random_state`.

The solution to all three is **mocking**: replacing the real implementation of a dependency with a controlled fake for the duration of the test.

---

## The Testing Setup

The project uses Python's built-in `unittest` and `unittest.mock`. No third-party testing libraries are required beyond `pytest` as the test runner.

```
tests/
├── __init__.py
├── test_data_ingestion.py
├── test_data_validation.py
├── test_data_transformation.py
├── test_model_trainer.py
└── test_model_evaluation.py
```

Run the full suite with:

```bash
python -m pytest tests/ -v
```

---

## Testing Data Ingestion

The `DataIngestion` component has two responsibilities: download a zip file from a URL, and extract it. Both involve I/O that we do not want to trigger in tests.

### The Config Object

Every component takes a typed config dataclass. In tests, we build one directly:

```python
from src.dscProject.entity.config_entity import DataIngestionConfig
from src.dscProject.components.data_ingestion import DataIngestion

def _make_config():
    return DataIngestionConfig(
        root_dir=Path("artifacts/data_ingestion"),
        source_URL="https://example.com/data.zip",
        local_data_file=Path("artifacts/data_ingestion/data.zip"),
        unzip_dir=Path("artifacts/data_ingestion"),
    )
```

Using a real config object (rather than a `MagicMock`) means we test the actual component initialisation path.

### Test: Download is skipped when file exists

```python
@mock.patch("src.dscProject.components.data_ingestion.os.path.exists", return_value=True)
@mock.patch("src.dscProject.components.data_ingestion.request.urlretrieve")
def test_download_file_skips_when_present(self, mock_retrieve, _mock_exists):
    DataIngestion(config=_make_config()).download_file()
    mock_retrieve.assert_not_called()
```

**Why this test matters:** The idempotency check in `download_file` is a behaviour that affects correctness. If someone accidentally removes the `os.path.exists` guard, re-running the pipeline will re-download data on every run. This test catches that regression.

**Key technique:** We patch `os.path.exists` at the module level where it is used (`src.dscProject.components.data_ingestion.os.path.exists`), not at the global level. This is a critical detail in Python mocking — you mock the name where it is *looked up*, not where it is *defined*.

### Test: Download is triggered when file is missing

```python
@mock.patch("src.dscProject.components.data_ingestion.os.path.exists", return_value=False)
@mock.patch("src.dscProject.components.data_ingestion.request.urlretrieve")
def test_download_file_fetches_when_missing(self, mock_retrieve, _mock_exists):
    mock_retrieve.return_value = ("data.zip", {})
    DataIngestion(config=_make_config()).download_file()
    mock_retrieve.assert_called_once_with(
        url=config.source_URL,
        filename=config.local_data_file,
    )
```

This verifies not just that `urlretrieve` was called, but that it was called with the *correct arguments* taken from the config object.

### Test: Extraction goes to the right directory

```python
@mock.patch("src.dscProject.components.data_ingestion.os.makedirs")
@mock.patch("src.dscProject.components.data_ingestion.zipfile.ZipFile")
def test_extract_zip_file_extracts_to_unzip_dir(self, mock_zipfile, _mock_makedirs):
    mock_zip = mock.MagicMock()
    mock_zipfile.return_value.__enter__ = mock.Mock(return_value=mock_zip)
    mock_zipfile.return_value.__exit__ = mock.Mock(return_value=False)

    config = _make_config()
    DataIngestion(config=config).extract_zip_file()

    mock_zip.extractall.assert_called_once_with(config.unzip_dir)
```

To mock a context manager (`with zipfile.ZipFile(...) as zip_ref`), you need to set `__enter__` and `__exit__` on the mock. `__enter__` returns the object that `zip_ref` is bound to; `__exit__` returns `False` to signal that exceptions should propagate normally.

---

## Testing Data Validation

The validation component has a known bug in the original code: when validation fails, it attempts `list - dict_keys`, which raises a `TypeError`. After the fix (`set(all_cols) - set(all_schema)`), we write a regression test that proves the bug is gone.

### Test: Valid columns return True

```python
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("src.dscProject.components.data_validation.pd.read_csv")
def test_returns_true_for_valid_columns(self, mock_read_csv, _mock_open):
    mock_read_csv.return_value = pd.DataFrame(columns=list(SCHEMA.keys()))
    result = DataValidation(config=_make_config()).validate_all_columns()
    self.assertTrue(result)
```

We mock `pd.read_csv` to return a DataFrame with the expected columns, and mock `builtins.open` to capture the status file write without touching the filesystem.

### Test: Regression — no TypeError on invalid columns

```python
@mock.patch("builtins.open", new_callable=mock.mock_open)
@mock.patch("src.dscProject.components.data_validation.pd.read_csv")
def test_invalid_columns_do_not_raise_type_error(self, mock_read_csv, _mock_open):
    mock_read_csv.return_value = pd.DataFrame(columns=["fixed acidity", "bad_col"])
    try:
        DataValidation(config=_make_config()).validate_all_columns()
    except TypeError:
        self.fail("validate_all_columns raised TypeError on invalid columns")
```

A regression test like this is valuable precisely because it documents a bug that existed. When someone reads this test name in six months, they know: this case was broken before, and it must stay fixed.

---

## Testing Data Transformation

The transformation stage splits a DataFrame and saves two CSV files. We test both the behaviour (two files are saved) and the paths (they are under the configured root directory).

### Test: Exactly two files are saved

```python
@mock.patch("src.dscProject.components.data_transformation.pd.read_csv")
def test_creates_two_csv_files(self, mock_read_csv):
    mock_read_csv.return_value = _sample_df()
    with mock.patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
        DataTransformation(config=_make_config()).train_test_split()
        self.assertEqual(mock_to_csv.call_count, 2)
```

We use `mock.patch.object` to patch the `to_csv` method on the `pd.DataFrame` class itself. This intercepts both calls (train set and test set) without writing any files.

### Test: Output paths are under root_dir

```python
@mock.patch("src.dscProject.components.data_transformation.pd.read_csv")
def test_output_paths_contain_root_dir(self, mock_read_csv):
    mock_read_csv.return_value = _sample_df()
    config = _make_config()
    saved_paths = []

    with mock.patch.object(pd.DataFrame, "to_csv",
                           side_effect=lambda path, **kw: saved_paths.append(str(path))):
        DataTransformation(config=config).train_test_split()

    for path in saved_paths:
        self.assertIn(str(config.root_dir), path)
```

Using `side_effect` with a lambda lets us capture what arguments were passed without preventing the mock from being called. We collect the paths and assert against them after the method returns.

---

## Testing Model Training

The model trainer is one of the more interesting components to test because we can make real assertions about the model object itself, not just about whether files were saved.

### Test: The saved model is actually an ElasticNet

```python
@mock.patch("src.dscProject.components.model_trainer.joblib.dump")
@mock.patch("src.dscProject.components.model_trainer.create_directories")
@mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
def test_saved_model_is_elasticnet(self, mock_read_csv, _mock_dirs, mock_dump):
    mock_read_csv.return_value = _sample_df()
    ModelTrainer(config=_make_config()).train_model()

    saved_model = mock_dump.call_args[0][0]
    self.assertIsInstance(saved_model, ElasticNet)
```

`mock_dump.call_args[0][0]` retrieves the first positional argument of the call to `joblib.dump`. Since `joblib.dump(model, model_path)` was called, `call_args[0][0]` is the actual fitted model object. We can then call `assertIsInstance` on it.

This test is valuable because it makes the contract explicit: this stage must produce a scikit-learn `ElasticNet` model.

### Test: Hyperparameters are applied correctly

```python
@mock.patch("src.dscProject.components.model_trainer.joblib.dump")
@mock.patch("src.dscProject.components.model_trainer.create_directories")
@mock.patch("src.dscProject.components.model_trainer.pd.read_csv")
def test_elasticnet_uses_configured_hyperparams(self, mock_read_csv, _mock_dirs, mock_dump):
    mock_read_csv.return_value = _sample_df()
    config = _make_config()
    ModelTrainer(config=config).train_model()

    saved_model: ElasticNet = mock_dump.call_args[0][0]
    self.assertAlmostEqual(saved_model.alpha, config.alpha)
    self.assertAlmostEqual(saved_model.l1_ratio, config.l1_ratio)
```

This test verifies that the hyperparameters from the config object actually reach the model. If someone hardcodes `ElasticNet(alpha=0.5)` instead of reading from config, this test fails.

---

## Testing Model Evaluation

The evaluation component has two distinct parts: the pure metric calculation (`eval_metrics`) and the MLflow integration (`log_into_mlflow`). We test them separately.

### Testing the Pure Function First

`eval_metrics` takes two numpy arrays and returns three numbers. It has no I/O. This is the easiest test to write:

```python
def test_perfect_predictions_give_zero_error(self):
    actual = np.array([5.0, 6.0, 7.0, 5.0, 6.0])
    rmse, mae, r2 = self.evaluator.eval_metrics(actual, actual.copy())
    self.assertAlmostEqual(rmse, 0.0)
    self.assertAlmostEqual(mae, 0.0)
    self.assertAlmostEqual(r2, 1.0)

def test_rmse_is_root_of_mse(self):
    actual = np.array([5.0, 6.0, 7.0])
    pred   = np.array([4.0, 5.0, 9.0])
    rmse, _, _ = self.evaluator.eval_metrics(actual, pred)
    expected_rmse = np.sqrt(np.mean((actual - pred) ** 2))
    self.assertAlmostEqual(rmse, expected_rmse, places=6)
```

The second test is particularly useful: it verifies the RMSE formula by computing the expected value independently. If the implementation ever switches to a different formula, this test catches it.

### Testing the MLflow Integration

```python
@mock.patch("src.dscProject.components.model_evaluation.mlflow")
@mock.patch("src.dscProject.components.model_evaluation.save_json")
@mock.patch("src.dscProject.components.model_evaluation.joblib.load")
@mock.patch("src.dscProject.components.model_evaluation.pd.read_csv")
def test_logs_all_three_metrics(self, mock_read_csv, mock_load,
                                _mock_save_json, mock_mlflow):
    mock_read_csv.return_value = self._make_test_df()
    mock_load.return_value.predict.return_value = np.array([5.0, 6.0, 5.0])
    mock_mlflow.get_tracking_uri.return_value = "https://dagshub.com/test/test.mlflow"

    ModelEvaluation(config=_make_config()).log_into_mlflow()

    logged_metrics = {call.args[0] for call in mock_mlflow.log_metric.call_args_list}
    self.assertSetEqual(logged_metrics, {"RMSE", "MAE", "R2_Score"})
```

We mock the entire `mlflow` module. `MagicMock` automatically supports context managers, so `with mlflow.start_run():` works without any extra setup.

The assertion uses a set comprehension over `call_args_list` to collect every metric name that was logged. `assertSetEqual` checks that exactly the right three metrics — and no others — were logged.

---

## The Three Techniques Summarised

After working through all five components, three techniques cover almost every testing scenario in an ML pipeline:

### 1. Patch the module-level name

```python
@mock.patch("src.dscProject.components.data_ingestion.os.path.exists", return_value=False)
```

Always patch where the name is *used*, not where it is *defined*. When `data_ingestion.py` does `import os`, it gets its own reference to `os`. Patching `os.path.exists` globally does not affect it — you must patch `data_ingestion.os.path.exists`.

### 2. Use `mock.patch.object` for instance methods

```python
with mock.patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
```

This patches a method on a class, which intercepts all calls from any instance of that class within the test scope.

### 3. Use `side_effect` to capture arguments

```python
side_effect=lambda path, **kw: saved_paths.append(str(path))
```

`side_effect` lets you run a function when the mock is called. Use it when you need to capture what arguments were passed, or to simulate an exception being raised.

---

## What These Tests Do Not Cover

Being honest about the limits of unit tests is as important as writing them.

**Integration between stages** — The unit tests verify each component in isolation. They do not verify that the output of data transformation is correctly consumed by model training. For that, you need integration tests that run adjacent stages together.

**Model accuracy** — No test here checks whether the trained model is accurate. Model accuracy is validated by the evaluation metrics logged to MLflow and tracked over time.

**Data distribution drift** — If the production data starts to look different from the training data, no unit test will catch it. That is the domain of data monitoring, which is a separate system.

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

Output:
```
tests/test_data_ingestion.py::TestDataIngestion::test_download_file_fetches_when_missing PASSED
tests/test_data_ingestion.py::TestDataIngestion::test_download_file_skips_when_present PASSED
tests/test_data_ingestion.py::TestDataIngestion::test_extract_zip_file_extracts_to_unzip_dir PASSED
tests/test_data_validation.py::TestDataValidation::test_invalid_columns_do_not_raise_type_error PASSED
tests/test_data_validation.py::TestDataValidation::test_returns_false_for_extra_columns PASSED
tests/test_data_validation.py::TestDataValidation::test_returns_true_for_valid_columns PASSED
tests/test_data_validation.py::TestDataValidation::test_status_written_to_file PASSED
tests/test_data_transformation.py::TestDataTransformation::test_creates_two_csv_files PASSED
tests/test_data_transformation.py::TestDataTransformation::test_output_paths_contain_root_dir PASSED
tests/test_data_transformation.py::TestDataTransformation::test_split_ratio_is_75_25 PASSED
tests/test_model_evaluation.py::TestEvalMetrics::test_imperfect_predictions_give_positive_error PASSED
tests/test_model_evaluation.py::TestEvalMetrics::test_perfect_predictions_give_zero_error PASSED
tests/test_model_evaluation.py::TestEvalMetrics::test_rmse_is_root_of_mse PASSED
tests/test_model_evaluation.py::TestLogIntoMlflow::test_logs_all_three_metrics PASSED
tests/test_model_evaluation.py::TestLogIntoMlflow::test_saves_metrics_json PASSED
tests/test_model_evaluation.py::TestLogIntoMlflow::test_sets_tracking_uri PASSED
tests/test_model_trainer.py::TestModelTrainer::test_elasticnet_uses_configured_hyperparams PASSED
tests/test_model_trainer.py::TestModelTrainer::test_model_is_saved PASSED
tests/test_model_trainer.py::TestModelTrainer::test_model_path_contains_model_name PASSED
tests/test_model_trainer.py::TestModelTrainer::test_saved_model_is_elasticnet PASSED

20 passed in 24.76s
```

---

## Key Takeaways

1. **ML pipeline code is testable.** Accuracy is not testable in unit tests. File I/O, path logic, parameter wiring, and API calls absolutely are.
2. **Patch where names are looked up, not where they are defined.** This is the single most common mocking mistake.
3. **Write regression tests for bugs you fix.** Name them after the bug so future readers know what they are protecting against.
4. **Separate pure functions from I/O-heavy functions.** `eval_metrics` is trivially easy to test because it has no side effects. `log_into_mlflow` requires four mocks. Design toward purity where you can.
5. **Assert on arguments, not just on call counts.** `assert_called_once()` tells you the function was called. `assert_called_once_with(url=..., filename=...)` tells you it was called correctly.

---

*This article is part of a three-part series. Part 1 covers pipeline architecture. Part 2 covers MLOps, serving, and security.*
