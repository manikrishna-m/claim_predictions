import pytest
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_config():
    """
    Fixture providing a sample configuration for the DataPreprocessor.
    """
    return {
        "input_columns": ["age", "income", "gender"],
        "target_column": "claim_status",
        "categorical_columns": ["gender"],
        "split": {
            "test_size": 0.2,
            "stratify": True,
            "random_state": 42
        }
    }

@pytest.fixture
def sample_data():
    """
    Fixture providing sample data as a pandas DataFrame for testing.
    """
    return pd.DataFrame({
        "age": [25, 30, 35, 40, np.nan, 50, 55, 60, np.nan, 70],
        "income": [50000, 60000, np.nan, 80000, 90000, 100000, np.nan, 120000, 130000, 140000],
        "gender": ["M", "F", "M", np.nan, "F", "M", "F", np.nan, "M", "F"],
        "claim_status": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "high_missing_col": [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    })

def test_initialization_success(sample_config):
    """
    Test successful initialization of DataPreprocessor and check attribute assignment.
    """
    preprocessor = DataPreprocessor(sample_config)
    assert preprocessor.input_columns == ["age", "income", "gender"]
    assert preprocessor.target_column == "claim_status"
    assert preprocessor.X_train is None
    assert preprocessor.X_test is None

def test_summarize_missing_values(sample_config, sample_data):
    """
    Test that the summarize_missing_values method returns a correct DataFrame with missing value info.
    """
    preprocessor = DataPreprocessor(sample_config)
    result = preprocessor.summarize_missing_values(sample_data)

    assert isinstance(result, pd.DataFrame)
    assert "high_missing_col" in result.index
    assert result.loc["high_missing_col", "Percent"] == 90.0
    assert "Types" in result.columns

def test_drop_high_missing_columns(sample_config, sample_data):
    """
    Test that columns with missing value percentages above threshold are dropped.
    """
    preprocessor = DataPreprocessor(sample_config)
    cleaned_data = preprocessor.drop_high_missing_columns(sample_data, threshold=50.0)

    assert "high_missing_col" not in cleaned_data.columns
    assert "age" in cleaned_data.columns  # Should be kept

def test_drop_high_missing_columns_no_summary(sample_config, sample_data):
    """
    Test that drop_high_missing_columns works without explicitly calling summarize_missing_values first.
    """
    preprocessor = DataPreprocessor(sample_config)
    cleaned_data = preprocessor.drop_high_missing_columns(sample_data)
    assert "high_missing_col" not in cleaned_data.columns

def test_cast_categorical_columns(sample_config, sample_data):
    """
    Test that specified categorical columns are cast to 'category' dtype.
    """
    preprocessor = DataPreprocessor(sample_config)
    processed_data = preprocessor.cast_categorical_columns(sample_data)

    assert str(processed_data["gender"].dtype) == "category"
    assert str(processed_data["age"].dtype) != "category"  # Not in categorical_columns

def test_cast_categorical_columns_missing_column(sample_config, sample_data):
    """
    Test that the function handles missing columns in categorical_columns gracefully.
    """
    modified_config = sample_config.copy()
    modified_config["categorical_columns"] = ["gender", "nonexistent_column"]
    preprocessor = DataPreprocessor(modified_config)
    processed_data = preprocessor.cast_categorical_columns(sample_data)

    assert str(processed_data["gender"].dtype) == "category"
    assert "nonexistent_column" not in processed_data.columns

def test_split_train_test_data(sample_config, sample_data):
    """
    Test that the train-test split works correctly with stratification.
    """
    preprocessor = DataPreprocessor(sample_config)
    X_train, X_test, y_train, y_test = preprocessor.split_train_test_data(sample_data)

    assert len(X_train) == 8  # 80% of 10 samples
    assert len(X_test) == 2   # 20% of 10 samples
    assert list(X_train.columns) == sample_config["input_columns"]
    assert y_train.name == "claim_status"

def test_split_train_test_data_no_stratify(sample_config, sample_data):
    """
    Test train-test split when stratification is disabled in the config.
    """
    modified_config = sample_config.copy()
    modified_config["split"]["stratify"] = False
    preprocessor = DataPreprocessor(modified_config)
    X_train, X_test, y_train, y_test = preprocessor.split_train_test_data(sample_data)

    assert len(X_train) == 8
    assert len(X_test) == 2

def test_split_train_test_data_default_params(sample_config, sample_data):
    """
    Test train-test split functionality when 'split' key is missing from the config.
    """
    modified_config = sample_config.copy()
    del modified_config["split"]
    preprocessor = DataPreprocessor(modified_config)
    X_train, X_test, y_train, y_test = preprocessor.split_train_test_data(sample_data)

    assert len(X_train) > 0
    assert len(X_test) > 0

def test_logging_behavior(sample_config, sample_data, caplog):
    """
    Test logging outputs for preprocessing steps using pytest's caplog fixture.
    """
    preprocessor = DataPreprocessor(sample_config)

    # Test summarize_missing_values logging
    preprocessor.summarize_missing_values(sample_data)
    assert "Missing value summary generated." in caplog.text

    # Test drop_high_missing_columns logging
    caplog.clear()
    preprocessor.drop_high_missing_columns(sample_data)
    assert "Dropped columns with >50.0% missing values: ['high_missing_col']" in caplog.text

    # Test cast_categorical_columns logging
    caplog.clear()
    preprocessor.cast_categorical_columns(sample_data)
    assert "Categorical columns cast to 'category' dtype." in caplog.text

    # Test split_train_test_data logging
    caplog.clear()
    preprocessor.split_train_test_data(sample_data)
    assert "Train-test split completed." in caplog.text
